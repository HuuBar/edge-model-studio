import concurrent.futures
import queue
import threading
import time
import sys
from typing import Callable, Any, List, Optional


class PipelineSentinel:
    """
    流式终止哨兵（Poison Pill）。
    用于在线性队列中向下游传递“数据已完结”的信号，优雅关停常驻消费线程。
    """
    pass


class PipelineStage:
    """
    流水线独立核心工作车间（Stage）
    """
    def __init__(self, name: str, action: Callable[[Any], Any], workers: int = 1, max_queue_size: int = 100):
        self.name = name
        self.action = action
        self.workers = workers
        # 带背压控制的输入阻塞队列
        self.in_queue = queue.Queue(maxsize=max_queue_size)
        self.out_queue: Optional[queue.Queue] = None

    def _worker_loop(self):
        """
        常驻工作线程的主循环体
        """
        while True:
            try:
                # 阻塞式获取上游灌入的数据
                data = self.in_queue.get()
                
                # 触发终止检查：如果收到哨兵信号，证明上游数据全部放完了
                if isinstance(data, PipelineSentinel):
                    # 把它原样送入下游，通知下游工作车间也准备收工
                    if self.out_queue:
                        self.out_queue.put(data)
                    self.in_queue.task_done()
                    break

                # 执行核心业务算子
                processed_data = self.action(data)

                # 将产出投递到下一级队列
                if self.out_queue and processed_data is not None:
                    self.out_queue.put(processed_data)

            except Exception as e:
                print(f"[CRITICAL] Stage '{self.name}' crashed with error: {e}", file=sys.stderr)
                # 发生灾难性异常时，向后追加哨兵，防止下游无限期死等阻塞
                if self.out_queue:
                    self.out_queue.put(PipelineSentinel())
                self.in_queue.task_done()
                raise e
            finally:
                self.in_queue.task_done()


class LinearQueuePipeline:
    """
    多级常驻阻塞队列流水线管理器（支持流式输入、自动背压防护）
    """
    def __init__(self):
        self.stages: List[PipelineStage] = []
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._futures: List[concurrent.futures.Future] = []

    def add_stage(self, stage: PipelineStage):
        """
        按顺序串联流水线车间
        """
        if self.stages:
            # 前一步的输出队列指向当前步的输入队列
            self.stages[-1].out_queue = stage.in_queue
        self.stages.append(stage)

    def start(self):
        """
        拉起全流水线，常驻工作线程在后台开始监听队列
        """
        if not self.stages:
            print("[WARN] No stages registered. Pipeline startup aborted.")
            return

        total_workers = sum(s.workers for s in self.stages)
        print(f"[INFO] Initializing Linear Pipeline with {len(self.stages)} stages ({total_workers} total threads).")
        
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=total_workers)
        self._futures.clear()

        # 逆序或者顺序拉起均可，让各层 Stage 的后台 Thread 开始挂起等数
        for stage in self.stages:
            for _ in range(stage.workers):
                future = self._executor.submit(stage._worker_loop)
                self._futures.append(future)
        print("[INFO] All backend storage/inference stages are warm and listening.")

    def put(self, item: Any):
        """
        作为外部生产者，向流水线的第一层源源不断地喂入原始数据。
        如果第一层处理不过来且队列满了，这里会自动阻塞，实施内存保护（背压）。
        """
        if not self.stages:
            raise RuntimeError("Pipeline is empty and cannot receive inputs.")
        self.stages[0].in_queue.put(item)

    def join(self, timeout_sec: Optional[float] = None):
        """
        输入结束，投放终止哨兵，并同步等待所有车间把残留数据全部洗完落盘。
        """
        if not self.stages:
            return

        print("[INFO] Injection finished. Sending PipelineSentinel downstream...")
        # 往第一层塞入哨兵
        self.stages[0].in_queue.put(PipelineSentinel())

        start_time = time.perf_counter()

        # 依次等待每个 Stage 清空自己内部累积的 task 计数
        for stage in self.stages:
            print(f"[WAITING] Flusing residual data inside stage: '{stage.name}'...")
            stage.in_queue.join()

        # 确认所有线程都优雅退出了
        if self._executor:
            self._executor.shutdown(wait=True)
            
        # 异常捕获检查
        for future in self._futures:
            if future.exception():
                print(f"[CRITICAL] Pipeline terminated with broken stage exception: {future.exception()}", file=sys.stderr)
                raise future.exception()

        elapsed = time.perf_counter() - start_time
        print(f"[SUCCESS] Stream Pipeline workflow accomplished smoothly. Flush time: {elapsed:.3f} s\n")