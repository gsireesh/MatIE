from dataclasses import dataclass
import os
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import List

import fastapi
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI

@dataclass
class MatIEEntity(BaseModel):
    id: str
    entity_type: str
    start: int
    end: int
    entity_string: str


relation_re = re.compile("R\d+\t(?P<r_type>.*) Arg1:(?P<arg1>T\d+) Arg2:(?P<arg2>T\d+)")


def generate_txt(self, working_folder, paragraph_text, key):
    file_path = os.path.join(
        working_folder, f"{key}.txt".replace("/", "_")
    )

    os.makedirs(working_folder, exists_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(paragraph_text)

    return paragraph_text


def run_matIE(working_folder):
    for dir_name in os.listdir(working_folder):
        if (
            not os.path.isdir(os.path.join(working_folder, dir_name))
            or "_original" in dir_name
        ):
            continue

        dir_path = os.path.join(working_folder, dir_name)
        input_folder = dir_path
        output_folder = dir_path

        process_files_multiprocess(
            input_folder,
            output_folder,
        )

def process_files_multiprocess(self, input_folder, output_folder):
    env_vars = os.environ.copy()
    env_vars["MODEL_DIR"] = self.NER_model_dir
    env_vars["VOCAB_DIR"] = self.vocab_dir
    env_vars["CUDA_VISIBLE_DEVICES"] = ""  # needs to fix later
    # bizarrely, taking out the `../ht-max` from these paths breaks something in MatIE
    env_vars["INPUT_DIR"] = os.path.join("./", input_folder.replace("//", "/"))
    env_vars["OUTPUT_DIR"] = os.path.join("./", output_folder.replace("//", "/"))
    env_vars["EXTRA_ARGS"] = ""

    subprocess.run(["chmod", "+x", self.decode_script], check=True)
    try:
        output = subprocess.check_output(
            self.decode_script,
            stderr=subprocess.STDOUT,
            shell=True,
            env=env_vars,
            cwd=self.matIE_directory,
        )
    except subprocess.CalledProcessError as e:
        print("Status : FAIL", e.returncode, e.output)

def parse_ann_content(ann_content):
    entities = []
    relations = []
    for line in ann_content.split("\n"):
        if line.startswith("T"):
            parts = line.split("\t")
            e_id = parts[0]
            e_type, e_start, e_end = parts[1].split()
            e_string = "\t".join(parts[2:])
            entity = MatIEEntity(e_id, e_type, int(e_start), int(e_end), e_string)
            entities.append(entity)
        elif line.startswith("R"):
            match = relation_re.fullmatch(line)
            relations.append(
                {
                    "relation_type": match.group("r_type"),
                    "arg1": match.group("arg1"),
                    "arg2": match.group("arg2"),
                }
            )

    return {"entities": entities, "relations": relations}


app.get("/")
def get_root():
    return {"hello": "world"}

@app.get("/status")
def get_status():
    return "Service is up!"

@app.post("/annotate_strings")
def annotate_strings(key_to_string: dict[str, str]) -> dict[str, dict[str, dict[str, str] | list[
    MatIEEntity]]]:
    print("Creating temporary input files")

    doc_temp_folder = TemporaryDirectory(prefix="matie_file_annotation_")

    input_paragraphs = {}

    for key, paragraph in key_to_string.items():
        paragraph_text = paragraph.replace("\n", " ")
        input_paragraphs[key] = generate_txt(doc_temp_folder.name, paragraph_text, key)

    print("Annotating temp files")
    run_matIE()

    print("Formatting annotated files...")

    return_content = dict()
    for key in input_paragraphs:
        folder_name = doc_temp_folder.name
        with open(
                os.path.join(folder_name, f"{key}.ann".replace("/", "_"))
        ) as f:
            annotated_content = parse_ann_content(f.read())
        with open(
                os.path.join(folder_name, f"{key}.txt".replace("/", "_"))
        ) as f:
            annotated_text = f.read()
            annotated_content["text"]  = annotated_text
        return_content[key] = annotated_content

    return return_content



