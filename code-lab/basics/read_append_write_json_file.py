"""A demo of json read, write and append in python.

Can be used for writing configuration data that is append based on some task.
"""

import json

class JsonReadWrite():
    def __init__(self, file_name):
        super(JsonReadWrite, self).__init__()
        self.file_to_read_and_write = file_name
        self.json_data = None

    def read_json(self):
        with open(self.file_to_read_and_write, 'r') as f:
            json_data = json.load(f)
            new_json_key = f"config_{len(json_data) + 1}"
            new_json_data = {new_json_key: f"data for {new_json_key}"}
            json_data.update(new_json_data)
            self.json_data = json_data

    def write_json(self):
        with open(self.file_to_read_and_write, 'w') as f:
            json.dump(self.json_data, f)

    def get_json(self):
        return self.json_data


if __name__ == '__main__':
    json_read_write = JsonReadWrite(file_name='read_append_write_json_file.json')
    json_read_write.read_json()
    print(json_read_write.get_json())
    json_read_write.write_json()