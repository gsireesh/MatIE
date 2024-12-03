# from dataclasses import dataclass
import logging
import os
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

NER_MODEL_DIR = "model"
VOCAB_DIR = "vpack_mat"
DECODE_SCRIPT = "./decode.sh"
WORKING_DIRECTORY = "matie_annotation"


@dataclass
class MatIEEntity:
    id: str
    entity_type: str
    start: int
    end: int
    entity_string: str


@dataclass
class MatIERelation:
    id: str
    relation_type: str
    arg1: str
    arg2: str


MatIEResponse = Dict[str, Dict[str, Union[str, List[MatIEEntity], List[MatIERelation]]]]


relation_re = re.compile(r"R(?P<r_id>\d+)\t(?P<r_type>.*) Arg1:(?P<arg1>T\d+) Arg2:(?P<arg2>T\d+)")


def generate_txt(tmp_folder, paragraph_text, key):
    file_path = os.path.join(
        tmp_folder, f"{key}.txt".replace("/", "_")
    )

    os.makedirs(tmp_folder, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(paragraph_text)

    return paragraph_text


def run_matie(working_folder):

    for dir_name in os.listdir(working_folder):
        if (
            not os.path.isdir(os.path.join(working_folder, dir_name))
            or "_original" in dir_name
        ):
            logger.error(f"{os.path.join(working_folder, dir_name)} not found!")
            continue

        dir_path = os.path.join(working_folder, dir_name)
        input_folder = dir_path
        output_folder = dir_path

        process_files_multiprocess(
            input_folder,
            output_folder,
        )


def process_files_multiprocess(input_folder, output_folder):
    env_vars = os.environ.copy()
    env_vars["MODEL_DIR"] = NER_MODEL_DIR
    env_vars["VOCAB_DIR"] = VOCAB_DIR
    env_vars["CUDA_VISIBLE_DEVICES"] = ""  # needs to fix later
    # bizarrely, taking out the `../ht-max` from these paths breaks something in MatIE
    env_vars["INPUT_DIR"] = os.path.join("./", input_folder.replace("//", "/"))
    env_vars["OUTPUT_DIR"] = os.path.join("./", output_folder.replace("//", "/"))
    env_vars["EXTRA_ARGS"] = ""

    subprocess.run(["chmod", "+x", DECODE_SCRIPT], check=True)
    try:
        logger.info("Running MatIE subprocess...")
        output = subprocess.check_output(
            DECODE_SCRIPT,
            stderr=subprocess.STDOUT,
            shell=True,
            env=env_vars,
        )
        logger.info("Done.")
    except subprocess.CalledProcessError as e:
        logger.error("Status : FAIL", e.returncode, e.output)


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
            relations.append(MatIERelation(
                id=match.group("r_id"),
                relation_type=match.group("r_type"),
                arg1=match.group("arg1"),
                arg2=match.group("arg2"),
            ))

    return {"entities": entities, "relations": relations}


app = FastAPI()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# StreamHandler for the console
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s")
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

@app.get("/")
def get_root():
    return {"hello": "world"}


@app.get("/status")
def get_status():
    return "Service is up!"


@app.post("/annotate_strings", response_model=MatIEResponse)
def annotate_strings(key_to_string: Dict[str, str]) -> MatIEResponse:
    logger.info("Creating temporary input files")

    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    doc_temp_folder = TemporaryDirectory(dir=WORKING_DIRECTORY, prefix="matie_file_annotation_")

    input_paragraphs = {}

    for key, paragraph in key_to_string.items():
        paragraph_text = paragraph.replace("\n", " ")
        input_paragraphs[key] = generate_txt(doc_temp_folder.name, paragraph_text, key)

    logger.info("Annotating temp files")
    run_matie(working_folder=WORKING_DIRECTORY)

    logger.info("Formatting annotated files...")

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
            annotated_content["text"] = annotated_text
        return_content[key] = annotated_content

    logger.info(return_content)
    return return_content
