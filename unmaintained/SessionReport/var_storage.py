import pickle
import os

class VariableOnDisk():
  '''
  Save and load variable on disk.
  '''

  def __init__(self, storage_path="./var_disk/"):
    os.makedirs(storage_path, exist_ok=True)
    self.storage_path = storage_path
  
  def set(self, variable_name, value):
    with open(os.path.join(self.storage_path, variable_name), 'wb') as f:
      pickle.dump(value, f)
  
  def get(self, variable_name):
    if os.path.exists(os.path.join(self.storage_path, variable_name)):
      with open(os.path.join(self.storage_path, variable_name), 'rb') as f:
        return pickle.load(f)
    else:
      raise NameError(f"name '{variable_name}' is not defined") # Same error when you try access variable that never defined.