#  Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import json
import argparse
def find_obj_files(directory, file_type):
    files_ = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(f'{file_type}'):
                files_.append(os.path.join(root, file))
    return files_

def save_to_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def main(directory_to_search, json_file_path, file_type) -> None:
    files = find_obj_files(directory_to_search, file_type)

    # 将.glb文件路径保存到JSON文件
    save_to_json(files, json_file_path)

    print(f"Found {len(files)} {file_type} files and saved to {json_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory_to_search",
        type= str,
        help="指定要遍历的目录",
    )
    parser.add_argument(
        "--json_file_path",
        type= str,
        help="指定要保存的JSON文件",
    )
    parser.add_argument(
        "--file_type",
        type= str,
        help="指定要检测的mesh后缀",
    )
    args, extras = parser.parse_known_args()
    main(args.directory_to_search, args.json_file_path, args.file_type)