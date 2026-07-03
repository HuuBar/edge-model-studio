import graphlib
import concurrent.futures
import time
import sys
from typing import Callable, List, Dict, Set, Any, Optional


class Task:
    """
    流水线原子任务封装
    """
    def __init__(self, name: str, action: Callable[..., Any], *args: Any, **kwargs: Any):
        self.name = name
        self.action = action
        self.args = args
        self.kwargs = kwargs

    def run(self, upstream_results: Dict[str, Any]) -> Any:
        """
        执行任务体。传入上游任务的返回值字典，供当前任务有选择地消费。
        """
        # 可以通过约定，让 action 自行决定是否消费 upstream_results
        # 这里为了通用性，直接执行原 action
        return self.action(*self.args, **self.kwargs)


class DepGraphPipeline:
    """
    基于有向无环图 (DAG) 的多线程并行流水线调度引擎
    """
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.graph: Dict[str, Set[str]] = {}
        self.results: Dict[str, Any] = {}  # 统一收集所有任务的返回值

    def add_task(self, task: Task, depends_on: Optional[List[str]] = None):
        """
        向图调度器注册任务及其前置依赖
        """
        if task.name in self.tasks:
            print(f"[WARN] Task '{task.name}' already registered. Overwriting.")
            
        dependencies = set(depends_on) if depends_on else set()
        self.tasks[task.name] = task
        self.graph[task.name] = dependencies

    def execute(self, max_workers: int = 4, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        """
        并行执行图任务。
        修复了原版在 get_ready() 为空且任务未完结时的 busy-loop 缺陷。
        """
        if not self.tasks:
            print("[WARN] No tasks registered in the pipeline.")
            return self.results

        print(f"[INFO] Pipeline Execution Initiated with {max_workers} workers.")
        
        # 统一清理上次运行的状态痕迹
        self.results.clear()

        # 初始化标准拓扑排序器（若存在循环依赖，此处会抛出 graphlib.CycleError）
        sorter = graphlib.TopologicalSorter(self.graph)
        sorter.prepare()

        start_time = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_node: Dict[concurrent.futures.Future, str] = {}

            while sorter.is_active():
                # 1. 尝试拉取当前依赖已完全解除的就绪节点
                ready_nodes = sorter.get_ready()

                if ready_nodes:
                    for node in ready_nodes:
                        task = self.tasks[node]
                        # 将全局上游结果集快照注入，赋予跨节点通信能力
                        future = executor.submit(task.run, self.results)
                        future_to_node[future] = node
                        print(f"[PROCESSING] Task '{node}' submitted to pool.")
                    # 提交完当前批次的所有就绪节点后，立即进入下一轮循环继续检查或步入下方的阻塞收尾
                    continue

                # 2. 关键防御：如果 get_ready() 返回空，但总体图仍处于 active 状态，
                # 说明当前没有新任务可以下发，所有可下发的任务都在线程池运行中。
                # 此时必须有条件地阻塞等待至少一个活跃任务完成，决不允许空转浪费 CPU。
                if future_to_node:
                    done, _ = concurrent.futures.wait(
                        future_to_node.keys(),
                        timeout=timeout_sec,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    # 3. 如果触发了整体超时控制，做收尾预警
                    if not done:
                        print(f"[CRITICAL] Pipeline allocation exceeded timeout limit of {timeout_sec}s.")
                        raise TimeoutError("Pipeline execution timed out.")

                    # 4. 依次回收这一批次执行完的 Future 对象
                    for future in done:
                        node = future_to_node.pop(future)
                        try:
                            # 捕获并拦截子线程异常
                            self.results[node] = future.result()
                            print(f"[SUCCESS] Task '{node}' finished successfully.")
                            
                            # 5. 反馈拓扑排序器，解开下游节点的依赖锁
                            sorter.done(node)
                        except Exception as e:
                            print(f"[CRITICAL] Pipeline aborted due to unexpected failure in task '{node}': {e}", file=sys.stderr)
                            # 迅速关闭线程池，不再接收新任务，止损
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise e

        elapsed = time.perf_counter() - start_time
        print(f"[SUCCESS] Pipeline workflow accomplished. Total time: {elapsed:.3f} s\n")
        return self.results