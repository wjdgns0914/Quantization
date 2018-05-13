from subprocess import Popen
import subprocess
def run_batch_file(file_path):
    Popen(file_path,creationflags=subprocess.CREATE_NEW_CONSOLE)
run_batch_file('./tt.bat')