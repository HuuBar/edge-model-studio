import json
import os.path
import sys
import unittest

import coverage

# 增加当前上级目录、上级目录子目录到 python解析器
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from unittestreport import TestRunner

# 测试类所在的文件夹绝对路径, 即 tests 目录的绝对路径
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# 工程所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 所有报告最终生成的路径，在工程目录下的 ut_cov_html文件夹中
REPORT_DIR = os.path.join(BASE_DIR, 'ut_cov_html')

# 耗时、成功率 报告路径
json_file_path = os.path.join(REPORT_DIR, "report.json")

if __name__ == "__main__":
    cov = coverage.Coverage(omit=['./*'])
    cov.erase()
    cov.start()
    # 找到所有测试文件
    suite = unittest.defaultTestLoader.discover(TESTS_DIR, pattern="test_*.py")

    # report.html 是指定的成功率, 耗时信息 的 html文件.
    test_runner = TestRunner(suite,
                            filename='report.html',
                            report_dir=REPORT_DIR,
                            tester="Test",
                            desc="Test测试报告",
                            templates=1
                            )
    result = test_runner.run()
    cov.stop()
    cov.save()
    res_report = dict(
        success=result.get('success'),
        errors=result.get('error'),
        failures=result.get('fail'),
        run_time=float(result.get('runtime')[:-1]),
        all_case=result.get('all'),
        cases=[],
    )

    # 覆盖率 html报告展示使用
    cov.html_report(directory=REPORT_DIR)
    # 覆盖率 xml文件、解析使用、生成增量覆盖率使用
    cov.xml_report(outfile=os.path.join(REPORT_DIR, 'coverage.xml'))
    if os.path.isfile(json_file_path):
        os.remove(json_file_path)
    print('============end tests======', json.dumps(res_report))
    print(json_file_path)
    with open(json_file_path, "w") as f:
        # 耗时、成功率解析的json文件
        f.write(json.dumps(res_report))