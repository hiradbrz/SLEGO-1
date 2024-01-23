def test225(input:str = 'Helwewlo'):
    return input

def list_lib():
  import subprocess
  import json
  data = subprocess.check_output(["pip", "list", "--format", "json"])
  parsed_results = json.loads(data)
  libs = [(element["name"], element["version"]) for 
  element in parsed_results]
  return libs

