import os
import json
from abc import ABC, abstractmethod


class IndexBase(ABC):
    @abstractmethod
    def create_index(self, key_name: str):
        pass

    @classmethod
    def store_index_id(cls, file_path, index):
        try:
            data = {}
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    data = json.load(file)
            data.update(index)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error storing index ID: {e}")

    @classmethod
    def load_index_id(cls, file_path, key_name: str):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            index_id = data.get(key_name)
            if index_id is None:
                print(f"Key '{key_name}' not found in {file_path}.")
            return index_id
        except FileNotFoundError:
            print(f"File {file_path} does not exist.")
        except json.JSONDecodeError:
            print(f"Invalid JSON in {file_path}.")
        except Exception as e:
            print(f"Error loading index ID: {e}")
        return None
