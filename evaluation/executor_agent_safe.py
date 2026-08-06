import json
import types
from typing import Optional, Callable, Dict
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile
from enum import Enum


class FResult(Enum):
    SAFE = 1  # validation returns okay
    FAILURE = 2  # validation contains error (something wrong with validation)
    ERROR = 3  # validation returns a potential error (look into)
    LLM_WEAKNESS = (
        4  # the generated input is ill-formed due to the weakness of the language model
    )
    TIMED_OUT = 10  # timed out, can be okay in certain targets


def execute_fuzz(completion: str, input_json, timeout: float,
                 completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    def get_exec_code(code, inputs_json):
        if isinstance(inputs_json, str):
            # Parse the inputs string from JSON
            # inputs = remove_json_prefix(inputs_json)
            inputs = json.loads(inputs)
        else:
            inputs = inputs_json
        # Prepare the Python script content
        t_scope = {}
        exec(code, t_scope)
        func_name = [obj for obj in t_scope.values(
        ) if isinstance(obj, types.FunctionType)][-1].__name__
        script_content = f"""
import json

{code}
    
def main():
    inputs = {inputs}

    try:
        result = {func_name}(**inputs)
        print(json.dumps({{"status": "success", "result": result}}))
    except Exception as e:
        print(json.dumps({{"status": "error", "error": str(e)}}))
        

if __name__ == "__main__":
    main()
"""
        return script_content, inputs, func_name

    def unsafe_execute(completion, input_json, timeout):
        with create_tempdir():
            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            # Disable functionalities that can make destructive changes to the test.
            # reliability_guard()
            code_to_run, input, func_name = get_exec_code(
                completion, input_json)

            # Construct the check program and run it.
            check_program = (
                code_to_run + "\n"
            )

            try:
                exec_globals = {}
                with time_limit(timeout):
                    exec(check_program, exec_globals)
                result = "passed"
            except TimeoutException as e:
                print(e)
                result = "timed out"
            except AssertionError as e:
                result = f"failed: AssertionError {e}"
            except Exception as e:
                result = f"failed: {e}"

            # Temporarily restore shutil functions for cleanup
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir
            return result, func_name
            # result_list.append(input)
            # result_list.append(result)
            # result_list.append(func_name)

    # manager = multiprocessing.Manager()
    # result_list = manager.list()
    # p = multiprocessing.Process(target=unsafe_execute(
    #     completion, input_json, timeout), args=result_list)
    # p.start()
    # p.join(timeout=0.5)

    # 调用unsafe_execute函数,返回input, result, func_name
    result, func_name = unsafe_execute(completion, input_json, timeout)
    return result, result == 'passed', func_name


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

