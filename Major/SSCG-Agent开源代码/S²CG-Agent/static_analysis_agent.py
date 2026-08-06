import csv
import json
import os
import subprocess
import uuid
import shutil
from abc import ABC, abstractmethod


class StaticAnalysisToolInterface(ABC):
    @abstractmethod
    def analyze_code(self, code: str) -> list:
        """
        对Python代码进行静态分析
        :param code: 需要分析的Python代码
        :return: 返回分析结果，列表形式
        """
        pass


class CodeQLStaticAnalyzer(StaticAnalysisToolInterface):
    def __init__(self, entry):
        self.entry = entry

    def extract_code(self, file_path, start_line, start_col, end_line, end_col):
        """
        从指定的文件中提取代码片段
        :param file_path: 源代码文件的路径
        :param start_line: 起始行号（从1开始）
        :param start_col: 起始列号（从1开始）
        :param end_line: 结束行号（从1开始）
        :param end_col: 结束列号（从1开始）
        :return: 提取的代码片段
        """
        # 读取源代码文件
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 提取目标行，记得行号从 1 开始，所以实际索引是 start_line-1
        target_lines = lines[start_line - 1:end_line]  # 提取起始行到结束行的部分

        if start_line == end_line:
            # 如果起始行和结束行相同，提取指定列范围
            # 列号从1开始，所以要减去1
            code_snippet = target_lines[0][start_col - 1:end_col]
        else:
            # 如果是多行代码，则需要处理第一行和最后一行
            code_snippet = ""

            # 第一行：从 start_col 到该行的结束
            # 第一行的代码从 start_col 开始
            code_snippet += target_lines[0][start_col - 1:]

            # 中间的行：直接添加
            for line in target_lines[1:-1]:
                code_snippet += line

            # 最后一行：从开头到 end_col
            code_snippet += target_lines[-1][:end_col]  # 最后一行的代码到 end_col 结束

        return code_snippet

    def analyze_code(self, code: str) -> list:
        # Use a unique temporary directory to avoid conflicts
        unique_id = str(uuid.uuid4())
        temp_dir = os.path.join("temp_codeql_analysis", unique_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        code_file_path = os.path.join(temp_dir, "temp_code.py")
        results_file_path = os.path.join(temp_dir, "results.csv")
        db_dir = os.path.join(temp_dir, "codeql-db")

        try:
            with open(code_file_path, "w") as temp_file:
                temp_file.write(code)

            # 创建 CodeQL 数据库
            create_db_command = [
                "codeql", "database", "create", db_dir, "--language=python", "--source-root=" + temp_dir,
            ]
            subprocess.run(create_db_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 使用与 baseline 一致的 suite 和 pack 路径
            pack_root = os.environ.get("CODEQL_PACK_ROOT", "scg-agent/codeql_query/e/.codeql-pack")
            suite_path = os.environ.get(
                "CODEQL_SUITE_PATH",
                os.path.join(
                    pack_root,
                    "codeql/python-queries/1.7.7/codeql-suites/python-security-and-quality.qls",
                ),
            )

            query_command = [
                "codeql", "database", "analyze", db_dir,
                "--format=csv",  # 设置格式为 csv
                "--output=" + results_file_path,  # 输出结果到 results.csv 文件
                suite_path,
                f"--search-path={pack_root}",
            ]
            
            # Try running analysis
            try:
                subprocess.run(query_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                # If analysis fails, return empty list
                return []

            # 读取 csv 格式的分析结果
            results = []
            if os.path.exists(results_file_path):
                with open(results_file_path, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)

                    for row in reader:
                        if len(row) < 9:
                            continue  # 如果该行不完整，则跳过

                        vulnerability = {
                            "name": row[0],  # 漏洞名称
                            "description": row[1],  # 漏洞描述
                            "severity": row[2],  # 漏洞类型（error/warning等）
                            "details": row[3],  # 漏洞详细描述
                            "file_path": row[4],  # 漏洞所在文件路径
                            "start_line": int(row[5]),  # 起始行号
                            "start_col": int(row[6]),  # 起始列号
                            "end_line": int(row[7]),  # 结束行号
                            "end_col": int(row[8])  # 结束列号
                        }
                        
                        # Fix path in vulnerability to be relative or absolute as needed
                        # But code_file_path is inside temp dir.
                        
                        vuln_code = self.extract_code(
                            code_file_path,
                            vulnerability["start_line"],
                            vulnerability["start_col"],
                            vulnerability["end_line"],
                            vulnerability["end_col"]
                        )
                        vulnerability["code"] = vuln_code

                        results.append(vulnerability)

            return results
            
        except Exception as e:
            print(f"CodeQL Analysis Error: {e}")
            return []
            
        finally:
            # Clean up the entire temporary directory
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Error cleaning up {temp_dir}: {e}")


class BanditStaticAnalysisTool(StaticAnalysisToolInterface):
    def __init__(self, entry):
        self.entry = entry

    def analyze_code(self, code: str) -> list:
        # Use unique temp file
        unique_id = str(uuid.uuid4())
        temp_file_name = f'temp_code_{unique_id}.py'
        
        try:
            with open(temp_file_name, 'w') as f:
                f.write(code)  # 将传入的代码写入临时文件

            # 使用 bandit 的绝对路径执行静态分析
            # Assume bandit is in path
            result = subprocess.run(
                ['/miniconda3/bin/bandit', '-f', 'json', temp_file_name], 
                capture_output=True,
                text=True,
                timeout=10
            )

            # 解析 bandit 输出的 JSON 格式结果
            analysis_results = []
            try:
                output = json.loads(result.stdout)
                # 提取 results 部分
                for issue in output.get("results", []):
                    analysis_results.append({
                        'test_id': issue.get('test_id'),
                        'issue': issue.get('issue_text'),
                        'severity': issue.get('issue_severity'),
                        'line_number': issue.get('line_number'),
                        'code': issue.get('code'),
                        'more_info': issue.get('more_info'),
                        'cwe_link': issue.get('issue_cwe', {}).get('link'),
                        'cwe_id': issue.get('issue_cwe', {}).get('id'),
                    })
            except json.JSONDecodeError:
                pass
                
            return analysis_results
            
        finally:
            # 删除临时文件
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)
