import os
import subprocess

def runScript(scriptName):
    print(f"Running {scriptName}.py.")
    result = subprocess.run(["python",f"{scriptName}.py"])
    print("Return code:", result.returncode)
    print("Output:", result.stdout)
    print("Error:", result.stderr)

runScript("train")
runScript("eval")

