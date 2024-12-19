from flask import Flask, request, render_template, jsonify,send_from_directory
print("importng for categories bp")
from categories import categories_bp
from documents import documents_bp
from rag import rag_bp

from dotenv import load_dotenv
import os
from app.dataExtraction.PDF.pdf_extract import extract,format_to_conversionOntology
import json
from app.GPT.gpt import summary_fn
from conn.mongodb import (
                              get_cat_by_id,
                              get_file_by_id,
                              add_extracted_formatted_data,
                              dashboard_data
                              )
from app.dataExtraction.Excel.excel_extract import extract_data_from_excel, format_row
from flask_socketio import SocketIO
import threading
from AzureVisionFormatter import AzureVisionBasedFormatter,AzureAssistedClaudeFormatter,AzureAssistedGPTFormatter
from app.dataExtraction.kyc import extract_text_for_kyc
from gpt_formatter import GPTFormatter,VisionExtractor
from claude_formatter import ClaudeEvaluator,ClaudeFormatter
from utils import replace_none_with_empty_string,replace_single_quotes
from medical_form_formatter import MedicalFormFormatter

import os
from pdf2image import convert_from_path
import cv2 as cv 
from PIL import Image

ALL_THREADS = []
print("Threads---------------------      ",ALL_THREADS)
load_dotenv()
from flask_cors import CORS  # Import CORS from flask_cors
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
# load necessary configurations
app = Flask(__name__, template_folder="./app/templates/", static_folder="./app/static")
app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER")
app.secret_key = os.getenv("SECRET_KEY")

app.register_blueprint(categories_bp)
app.register_blueprint(documents_bp)
app.register_blueprint(rag_bp)


CORS(app)
socketio = SocketIO(app)  # Specify the path, e.g., '/test/socket.io'
os.makedirs("indices",exist_ok=True)
os.makedirs("multiple_indices", exist_ok=True)
from langchain.output_parsers.json import SimpleJsonOutputParser

#   --------------- Flask API Views ---------------------

# /////////////////////////////////////////// TEMPLATE RENDERES /////////////////////////////////////////
@app.route("/")
@app.route("/dashboard")
def index():
    """
    Render Dashboard Page
    """
    total_documents,processed,inprogress,not_processed=dashboard_data()
    simple_agents=len(os.listdir("indices"))
    query_agents=len(os.listdir("multiple_indices"))
    total_agents=simple_agents+query_agents
    # categories = get_all_categories()
    return render_template(
        "dashboard.html",
            total_documents= total_documents,
            processed=processed,
            error= not_processed,
            inprogress=inprogress,
            total_agents=total_agents
    )


@app.route('/get_extracted_data/<id>/', methods=["POST"])
def start_background_task(id):
    # Start the background task
    print("args-------------------------",request.args)
    use_format = request.args.get("useFormat")
    thread = threading.Thread(target=extract_data, args=(id,use_format))
    thread.start()
    ALL_THREADS.append(thread)
    print_prcesses()

    # Respond immediately to the client
    return jsonify({'message': 'Background task started'})

def replace_none_with_empty_string(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if value is None:
                data[key] = ''
            else:
                replace_none_with_empty_string(value)
    elif isinstance(data, list):
        for item in data:
            replace_none_with_empty_string(item)

def replace_single_quotes(obj):
    if isinstance(obj, dict):
        return {key: replace_single_quotes(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [replace_single_quotes(item) for item in obj]
    elif isinstance(obj, str):
        return obj.replace("'", "")
    else:
        return obj

def remove_blank_null(json_data):
    cleaned_data = {}

    for key, value in json_data.items():
        if value not in ["", None]:
            if isinstance(value, dict):
                cleaned_value = remove_blank_null(value)
                if (
                    cleaned_value
                ):  # If the nested dictionary is not empty after cleaning
                    cleaned_data[key] = cleaned_value
            elif isinstance(value, list):
                print(value)
                cleaned_value = [item for item in value if item not in ["", None]]
                if cleaned_value:  # If the list is not empty after cleaning
                    cleaned_data[key] = cleaned_value
            else:
                cleaned_data[key] = value

    return cleaned_data

def crop_QR(image_path, coordinates, doc_id):
    os.makedirs("QR", exist_ok=True)
    image = Image.open(image_path)
    print(coordinates.shape)
    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    x3, y3 = coordinates[2]
    x4, y4 = coordinates[3]
    print("---------------------")
    cropped_image = image.crop((x1, y1, x3, y3))
    cropped_image.save(f"QR/{doc_id}.png")
    print("Image cropped and saved successfully.")


def get_qr(image_path):
    print("------------QR image ------------------",image_path)
    try:
        image = cv.imread(image_path)
        qrCodeDetector = cv.QRCodeDetector()
        decodedText, points, _ = qrCodeDetector.detectAndDecode(image)
        print("text------", decodedText)
        print("points--------", points)
        return decodedText, points
    except:
        return None ,None

def pdf_to_images(pdf_path,name):
    # Convert PDF to a list of PIL Image objects
    images = convert_from_path(pdf_path)
    f_images = []
    os.makedirs("cropped_images", exist_ok=True)

    for i, image in enumerate(images):
        # Save the image as PNG format
        image_path = f"cropped_images/{name}_{i+1}.png"
        f_images.append(image_path)
        image.save(image_path, "PNG")
        print(f"Page {i + 1} converted to image: {image_path}")
    return f_images


def extract_values_from_fomatted_data(formatted_json):
    values = []
    for val in formatted_json.values():
        if isinstance(val, dict):
            values.extend(extract_values_from_fomatted_data(val))
        else:
            values.append(val)

    return values


def get_confidence(values, confidence_scores):
    confidence_dict = {}

    for string in values:
        total_confidence = 0
        count = 0
        s = str(string)
        for word in s.split():
            n_word = word.replace(",", "")  # Replace commas with empty strings
            for d in confidence_scores:
                if n_word in d:
                    total_confidence += d[n_word]
                    count += 1
                    break
        if count > 0:
            avg_confidence = total_confidence / count
            confidence_dict[s] = avg_confidence
        else:
            # If no confidence score found for any word in the string, assign a default confidence score
            confidence_dict[s] = 0.00

    return confidence_dict


def get_confidence_using_GPT(formatted_json, confidence_json):
    MODEL_NAME = "gpt-3.5-turbo-16k"
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)

    template = """{prompt}
        The  Formatted_JSON object is :{formatted_json}
        The  confidence_scores_JSON object is :{confidence_json}
        
        \n\nThe Final JSON result is :"""
    prompt = """
    You are an expert in understanding json objects, your objective is to understand the provided json objects. The provided json objects are Formatted_JSON and Confidence_scores_JSON.

    Now, Identify the values in the Formatted_JSON object provided and also identify its corresponding confidence score from the Confidence_scores_JSON.
    If there are multiple occurences of same word as a sub string or any other form, identify the value in confidence json with respect to the context of the data. 
    If the value is combination of multiple words then the confidence score is average of all the confidenece scores of the words in the value.

    Finally return a Json with the value from formatted_JSON as a key and its corresponding confidence score as value.
    """
    try:
        print("#" * 20, "in side try")
        prompt_template = PromptTemplate(
            input_variables=["prompt", "formatted_json", "confidence_json"],
            template=template,
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="result")
        output = llm_chain.invoke(
            {
                "prompt": prompt,
                "formatted_json": formatted_json,
                "confidence_json": confidence_json,
            }
        )
        print(output["result"])
        return output["result"]
    except Exception as e:
        print("error in formatting is-----------", {e})
        return None


def remove_empty_fields(data):
    if isinstance(data, dict):
        return {
            k: remove_empty_fields(v)
            for k, v in data.items()
            if v and remove_empty_fields(v)
        }
    elif isinstance(data, list):
        return [remove_empty_fields(v) for v in data if v and remove_empty_fields(v)]
    else:
        return data


def remove_extra_quotes(dictionary):
    modified_dict = {}
    for key, value in dictionary.items():
        # Remove extra quotes from the key
        modified_key = key.replace('"', "").replace("'", "")

        # If the value is a nested dictionary, recursively process it
        if isinstance(value, dict):
            modified_value = remove_extra_quotes(value)
        elif isinstance(value, list):
            # If the value is a list, recursively process each element
            modified_value = [
                remove_extra_quotes(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            # Remove extra quotes from the value if it's a string
            modified_value = (
                value.replace('"', "").replace("'", "")
                if isinstance(value, str)
                else value
            )

        # Update the modified dictionary with the new key-value pair
        modified_dict[modified_key] = modified_value

    return modified_dict


# This is a backgorund function, the socket however listens to the response of the method and updates the data on the UserScreen
def extract_data(id, use_format=None):
    '''
    Method to extract the formatted and raw text based on the ontology provided from the file uploaded
    '''
    doc_file = get_file_by_id(id)
    doc_cat = get_cat_by_id(doc_file['category_id'])
    print("***************************************",doc_file)
    print("***************************************",doc_cat)
    category_type=doc_cat["file_type"]
    file_path = doc_file['file_path']
    print("--------------extarction of file path is -----------",file_path)

    if category_type=="pdf":
        print("inside pdf block")
        raw_data = extract(file_path, doc_cat['extraction_ontology'])
        print("^"*100)
        print(raw_data)
        # if use_format:
        add_extracted_formatted_data(id, {"extracted_text": raw_data})
        formatted = (
            format_to_conversionOntology(raw_data, doc_cat['conversion_ontology'], doc_cat['prompt_instructions']))
        if formatted is not None:
            data = {
                "formatted_data": formatted
            }
            add_extracted_formatted_data(id, data)
            socketio.emit('update', {'message': 'Background task completed!', 'result': data, 'data': raw_data,
                                     'formatted': str(formatted), "status": True})
            ALL_THREADS.pop(0)
        else:
            data = {
                "formatted_data": "Error formatting data"
            }

            add_extracted_formatted_data(id, data)
            socketio.emit('update',
                          {'message': 'Background task could not be completed!', 'result': data, 'data': raw_data,
                           'formatted': str(formatted), "status": False})
            ALL_THREADS.pop(0)

    elif category_type == "xlsx":
        excel_data = extract_data_from_excel(file_path)
        row = excel_data[0]

        add_extracted_formatted_data(id, {"extracted_text": row})
        row_formatted = format_row(row=row, format_string=doc_cat['conversion_ontology'],
                                   prompt=doc_cat['prompt_instructions'])
        if row_formatted is not None:
            data = {
                "formatted_data": row_formatted
            }
            add_extracted_formatted_data(id, data)
            socketio.emit('update',
                          {'message': 'Background task completed!', 'result': data, "data": data, "status": True},
                          namespace='/test')
            ALL_THREADS.pop(0)
        else:
            data = {
                "formatted_data": "Error Formatting data"
            }
            add_extracted_formatted_data(id, data)

            socketio.emit('update', {'message': 'Background task could not be completed!', 'result': data, "data": data,
                                     "status": False}, namespace='/test')
            ALL_THREADS.pop(0)

    elif category_type == "kyc":
        text=extract_text_for_kyc(file_path)
        add_extracted_formatted_data(id, {"extracted_text": text.replace("\n"," ")})
        if use_format:
            print("with formatt ","*^"*30)
            formatted = summary_fn(input_text=text, prompt=doc_cat["prompt_instructions"],format_string=doc_cat["conversion_ontology"])
        else:
            print("without formatt ", "*^" * 30)
            formatted = summary_fn(
                input_text=text, prompt=doc_cat["prompt_instructions"]
            )
        print("*"*20,type(formatted))
        try:
            formatted=json.loads(formatted)
            print("formated is ----------",type(formatted))
        except json.JSONDecodeError as e:
            print("-----------     ERROR  -----",e)
        if formatted is not None:
            data = {"formatted_data": formatted}
            add_extracted_formatted_data(id, data)
            socketio.emit(
                "update",
                {
                    "message": "Background task completed!",
                    "result": data,
                    "data": text,
                    "formatted": formatted,
                    "status": True,
                },
            )
            ALL_THREADS.pop(0)
        else:
            data = {"formatted_data": "Error formatting data"}

            add_extracted_formatted_data(id, data)
            socketio.emit(
                "update",
                {
                    "message": "Background task could not be completed!",
                    "result": data,
                    "data": text,
                    "formatted": formatted,
                    "status": False,
                },
            )
            ALL_THREADS.pop(0)
            print("completed------------------------------------------------------")

        pass

    elif category_type == "scanned":
        print("inside scanned------------------")
        doc_formatter=GPTFormatter(uploaded_image_path=file_path)
        data=doc_formatter.ocr_text
        add_extracted_formatted_data(id, {"extracted_text": data.replace("\n", " ")})

        if use_format:
            formatted_data=doc_formatter.format(format_structure=doc_cat["conversion_ontology"])
            print("with formatt ","*^"*30)
        else:
            print("without formatt ", "*^" * 30)
            formatted_data = doc_formatter.format(
                format_structure=doc_cat["conversion_ontology"]
            )
        print("*"*20,type(formatted_data))
        try:
            if not isinstance(formatted_data,dict):
                formatted=json.loads(formatted_data)
            else:
                formatted=formatted_data
            print("formated is ----------",type(formatted))
        except json.JSONDecodeError as e:
            print("-----------     ERROR  -----",e)

        if formatted is not None:
            data = {"formatted_data": formatted}
            add_extracted_formatted_data(id, data)
            socketio.emit(
                "update",
                {
                    "message": "Background task completed!",
                    "result": data,
                    "formatted": formatted,
                    "status": True,
                },
            )
            ALL_THREADS.pop(0)
        else:
            data = {"formatted_data": "Error formatting data"}

            add_extracted_formatted_data(id, data)
            socketio.emit(
                "update",
                {
                    "message": "Background task could not be completed!",
                    "result": data,
                    "formatted": None,
                    "status": False,
                },
            )
            ALL_THREADS.pop(0)
            print("completed------------------------------------------------------")

    elif category_type == "image":
        #     if use_format == "yes":
        #         print("with formatt ", "*^" * 30)
        #         prompt = f"""{doc_cat["prompt_instructions"]}\nThe final expected json format is {doc_cat["conversion_ontology"]}\n The final Json response is:"""
        #         image_extractor = VisionExtractor(image_path=file_path, prompt=prompt)
        #         formatted = image_extractor.extract2()
        #     else:
        #         print("without formatt ", "*^" * 30)
        #         prompt = doc_cat["prompt_instructions"]
        #         image_extractor = VisionExtractor(image_path=file_path, prompt=prompt)
        #         formatted = image_extractor.extract2()
        #     print("*" * 20, type(formatted))
        #     try:
        #         if isinstance(formatted, dict):
        #             pass
        #         else:
        #             formatted = json.loads(formatted)
        #         print("formated is ----------", type(formatted))
        #     except json.JSONDecodeError as e:
        #         print("-----------     ERROR  -----", e)
        #     if formatted is not None:
        #         replace_none_with_empty_string(formatted)
        #         formatted = replace_single_quotes(formatted)
        #         data = {"formatted_data": formatted}
        #         add_extracted_formatted_data(id, data)
        #         socketio.emit(
        #             890 - "update",
        #             {
        #                 "message": "Background task completed!",
        #                 "result": data,
        #                 "formatted": formatted,
        #                 "status": True,
        #             },
        #         )
        #         ALL_THREADS.pop(0)
        #     else:
        #         data = {"formatted_data": "Error formatting data"}

        #         add_extracted_formatted_data(id, data)
        #         socketio.emit(
        #             "update",
        #             {
        #                 "message": "Background task could not be completed!",
        #                 "result": data,
        #                 "formatted": formatted,
        #                 "status": False,
        #             },
        #         )
        #         ALL_THREADS.pop(0)
        #         print("completed------------------------------------------------------")
        pass

    elif category_type == "medicalForm":
        print("with formatt ", "*^" * 30)
        med_formatter = MedicalFormFormatter(
            image_url=file_path,
            doc_id=id,
            prompt=doc_cat["prompt_instructions"],
            conversion_ontology=doc_cat["conversion_ontology"]
        )
        formatted = med_formatter.start_eval()
        print("*" * 20, type(formatted))

        socketio.emit(
            890 - "update",
            {
                "message": "Background task completed!",
                "result": data,
                "formatted": formatted,
                "status": True,
            },
        )
        print("completed-- eval----------------------------------------------------",type(formatted))

    elif category_type == "azureVision":
        print("with formatt ", "*^" * 30)
        print("with formatt ", "*^" * 30)
        print("----------", file_path)
        print(file_path.split("\\")[-1])
        file_name = file_path.split("\\")[-1]
        has_QR=False
        file_name = file_path.split("\\")[-1]
        if file_name.split(".")[-1] == "pdf":
            name=file_name.split(".")[0]
            images=pdf_to_images(pdf_path=file_path,name=name)
            file_path=images[0]
            QR_text,points=get_qr(image_path=file_path)
            file_name = file_path.split("\\")[-1]
            if points is not None:
                has_QR = True
                # crop_QR(image_path=file_path, coordinates=points[0], doc_id=id)
            url_path = f"https://intellexi.walkingtree.tech/pdfimages/{file_name}"
        else:
            QR_text, points=get_qr(image_path=file_path)
            if points is not None:
                has_QR = True
                name = file_name.split(".")[0]
                # crop_QR(image_path=file_path,coordinates=points[0],doc_id=id)
            url_path = f"https://intellexi.walkingtree.tech/images/{file_name}"

        print(url_path)
        azureFormtter = AzureVisionBasedFormatter(image_url=url_path)
        extracted_text, confidence_scores = azureFormtter.extract()
        print("*" * 20, "Extarcted_text----------", extracted_text)
        print("*" * 20, "confidence Scroes----------", confidence_scores)
        claude_formatter = AzureAssistedClaudeFormatter(
            image_url=file_path,
            doc_id=id,
            prompt_instructions=doc_cat["prompt_instructions"],
            conversion_ontology=doc_cat["conversion_ontology"],
            vision_text=extracted_text,
        )
        formatted_data = claude_formatter.extract_using_claude()
        replace_none_with_empty_string(formatted_data)
        formatted_data = replace_single_quotes(formatted_data)
        print("*" * 20, type(formatted_data))

        try:
            if not isinstance(formatted_data, dict):
                formatted = json.loads(formatted_data)
            else:
                formatted = formatted_data
            print("formated is ----------", type(formatted))
        except json.JSONDecodeError as e:
            print("-----------     ERROR  -----", e)
            index = formatted.find("{")
            if index != -1:
                formatted = formatted_data[index:]
                print("Substring:", formatted)
            else:
                print("Character '{' not found in the string.")
            parser=SimpleJsonOutputParser()
            parsed_json = parser.parse(formatted)
            print("     **********  PArsing         JSON    OBJECT  **********")
            if isinstance(parsed_json, (dict, list)):
                formatted = parsed_json
            else:

                formatted = formatted

        if isinstance(formatted,(dict,list)):
            formatted=remove_blank_null(formatted)
        print("****************8Removed the blanks*************")
        print(formatted)
        formatted_values=extract_values_from_fomatted_data(formatted_json=formatted)
        print("*****************    values      ****************")
        print(formatted_values)
        confidence_scores_list=list(confidence_scores.values())
        print("********************8scores((*******************))",confidence_scores)
        json_with_confidence = get_confidence_using_GPT(
            formatted_json=formatted, confidence_scores=confidence_scores_list
        )

        try:
            if not isinstance(json_with_confidence, dict):
                json_loaded_json_with_confidence = json.loads(json_with_confidence)
            else:
                json_loaded_json_with_confidence = json_with_confidence
            print("formated o1p is ----------", type(json_loaded_json_with_confidence))
        except json.JSONDecodeError as e:
            print("-----------     ERROR   op 1-----", e)
            index = json_loaded_json_with_confidence.find("{")
            if index != -1:
                json_loaded_json_with_confidence = json_loaded_json_with_confidence[index:]
                print("op   Substring:", json_loaded_json_with_confidence)
            else:
                print("op  Character '{' not found in the string.")
            parserop = SimpleJsonOutputParser()
            parsed_json_op = parserop.parse(json_loaded_json_with_confidence)
            print("     **********  PArsing         JSON    CONFIDENCE  OBJECT  **********")
            if isinstance(parsed_json_op, (dict, list)):
                json_loaded_json_with_confidence= parsed_json_op
            else:

                json_loaded_json_with_confidence = json_loaded_json_with_confidence
        if formatted is not None:
            data = {
                "formatted_data": formatted,
                "extracted_text": extracted_text,
                "confidence_scores": confidence_scores,
                "formatted_with_confidence": json_loaded_json_with_confidence,
                "has_qr": has_QR,
                "qr_content": QR_text,
            }
            add_extracted_formatted_data(id, data)
            socketio.emit(
                "update",
                {
                    "message": "Background task completed!",
                    "result": data,
                    "formatted": formatted,
                    "status": True,
                },
            )
            ALL_THREADS.pop(0)
        else:
            data = {
                "formatted_data": "Error formatting data",
                "extarcted_text": extracted_text,
                "confidence_scores": confidence_scores,
                "has_qr":has_QR,
                "qr_content":QR_text
            }

            add_extracted_formatted_data(id, data)
            socketio.emit(
                "update",
                {
                    "message": "Background task could not be completed!",
                    "result": data,
                    "formatted": None,
                    "status": False,
                },
            )
            ALL_THREADS.pop(0)
        print("completed------------------------------------------------------")

    elif category_type == "azureOpenAI":
        print("with formatt ", "*^" * 30)
        print("with formatt ", "*^" * 30)
        print("----------", file_path)
        print(file_path.split("\\")[-1])
        file_name = file_path.split("\\")[-1]
        has_QR=False
        file_name = file_path.split("\\")[-1]
        if file_name.split(".")[-1] == "pdf":
            name=file_name.split(".")[0]
            images=pdf_to_images(pdf_path=file_path,name=name)
            image_file_path=images[0]
            QR_text,points=get_qr(image_path=image_file_path)
            image_file_name=image_file_path.split("/")[-1]
            image_path=image_file_path
            print(image_file_path,image_file_name)
            print("-------------filename1-----------",image_file_name)
            if points is not None:
                has_QR = True

                # crop_QR(image_path=file_path, coordinates=points[0], doc_id=id)
            url_path = f"https://intellexi.walkingtree.tech/pdfimages/{image_file_name}"
        else:
            QR_text, points=get_qr(image_path=file_path)
            if points is not None:
                has_QR = True
                name = file_name.split(".")[0]
                # crop_QR(image_path=file_path,coordinates=points[0],doc_id=id)
            image_path=file_path
            print("-------------filename2-----------",file_name)
            url_path = f"https://intellexi.walkingtree.tech/images/{file_name}"

        print("url--------",url_path)
        azureFormtter = AzureVisionBasedFormatter(image_url=url_path)
        extracted_text, confidence_scores = azureFormtter.extract()
        print("*" * 20, "Extarcted_text----------", extracted_text)
        print("*" * 20, "confidence Scroes----------", confidence_scores)
        print("\n\n\n",image_path,"-------image-----")
        gpt_formatter = AzureAssistedGPTFormatter(
            image_url=image_path,
            doc_id=id,
            prompt_instructions=doc_cat["prompt_instructions"],
            conversion_ontology=doc_cat["conversion_ontology"],
            vision_text=extracted_text,
        )
        formatted_data = gpt_formatter.extract()
        print(formatted_data)
        print("----------------AFTER GPT Vision---------------")
        print(formatted_data)
        replace_none_with_empty_string(formatted_data)
        formatted_data = replace_single_quotes(formatted_data)
        print("*" * 20, type(formatted_data))

        try:
            if not isinstance(formatted_data, dict):
                formatted = json.loads(formatted_data)
            else:
                formatted = formatted_data
            print("Formatted data type:", type(formatted))
        except json.JSONDecodeError as decode_error:
            print("JSON Decode Error:", decode_error)
            try:
                parser = SimpleJsonOutputParser()
                parsed_json = parser.parse(formatted_data)
                print("Parsed JSON object successfully")
                if isinstance(parsed_json, (dict, list)):
                    formatted = parsed_json
                else:
                    index = formatted_data.find("{")
                    if index != -1:
                        formatted = formatted_data[index:]
                        print("Substring:", formatted)
                    else:
                        print("Character '{' not found in the string.")
                        formatted = formatted_data
            except Exception as parse_error:
                print("Parsing Error:", parse_error)
                formatted = None  # Reset formatted if parsing fails
        try:
            if isinstance(formatted,dict):
                formatted=remove_empty_fields(formatted)
            else:
                try:
                    parser = SimpleJsonOutputParser()
                    parsed_json = parser.parse(formatted_data)
                    print("Parsed JSON object successfully")
                    if isinstance(parsed_json, (dict, list)):
                        formatted = parsed_json
                    else:
                        index = formatted_data.find("{")
                        if index != -1:
                            formatted = formatted_data[index:]
                            print("Substring:", formatted)
                        else:
                            print("Character '{' not found in the string.")
                            formatted = formatted_data                  
                except Exception as e:
                    print("EXCEPTIO ----2222222222",e)
                    formatted=formatted_data
        except Exception as e:
            print("exception 3333333333333333333333 -------",e)
            print(type(formatted) ,type(formatted_data))
            formatted=None
        if formatted:
            formatted=remove_empty_fields(formatted)
            formatted = remove_extra_quotes(formatted)
            confidence_scores=remove_extra_quotes(confidence_scores)
        print("****************8Removed the blanks and extra quotes*************")
        print(formatted)
        formatted_values=extract_values_from_fomatted_data(formatted_json=formatted)
        print("*****************    values      ****************")
        print(formatted_values)
        confidence_scores_list = list(confidence_scores.values())
        print("********************8scores((*******************))",confidence_scores)

        json_with_confidence = get_confidence_using_GPT(
        formatted_json=formatted, confidence_json=confidence_scores_list
    )

        try:
            if not isinstance(json_with_confidence, dict):
                json_loaded_json_with_confidence = json.loads(json_with_confidence)
            else:
                json_loaded_json_with_confidence = json_with_confidence
            print("formated o1p is ----------", type(json_loaded_json_with_confidence))
        except json.JSONDecodeError as e:
            print("-----------     ERROR   op 1-----", e)
            index = json_loaded_json_with_confidence.find("{")
            if index != -1:
                json_loaded_json_with_confidence = json_loaded_json_with_confidence[index:]
                print("op   Substring:", json_loaded_json_with_confidence)
            else:
                print("op  Character '{' not found in the string.")
            parserop = SimpleJsonOutputParser()
            parsed_json_op = parserop.parse(json_loaded_json_with_confidence)
            print("     **********  PArsing         JSON    CONFIDENCE  OBJECT  **********")
            if isinstance(parsed_json_op, (dict, list)):
                json_loaded_json_with_confidence= parsed_json_op
            else:
                json_loaded_json_with_confidence = json_loaded_json_with_confidence

        if formatted is not None:
            data = {
                "formatted_data": formatted,
                "extracted_text": extracted_text,
                "confidence_scores": confidence_scores,
                "formatted_with_confidence": json_loaded_json_with_confidence,
                "has_qr": has_QR,
                "qr_content": QR_text,
                "image_path":image_path
            }
            add_extracted_formatted_data(id, data)
            socketio.emit(
                "update",
                {
                    "message": "Background task completed!",
                    "result": data,
                    "formatted": formatted,
                    "status": True,
                },
            )
            ALL_THREADS.pop(0)
        else:
            data = {
                "formatted_data": "Error formatting data",
                "extarcted_text": extracted_text,
                "confidence_scores": confidence_scores,
                "has_qr": has_QR,
                "qr_content": QR_text,
                "image_path": image_path,
            }

            add_extracted_formatted_data(id, data)
            socketio.emit(
                "update",
                {
                    "message": "Background task could not be completed!",
                    "result": data,
                    "formatted": None,
                    "status": False,
                },
            )
            ALL_THREADS.pop(0)
        print("completed------------------------------------------------------")


@app.route("/images/<path:filename>")
def get_image(filename):
    print("----------------",filename)
    if filename.endswith(".pdf"):
        print("get iage of the pdf",filename,filename.split("\\")[-1])
        image_name = filename.split("\\")[-1]
        print("image_______",image_name)
        image_path=image_name.split(".")[0]
        print("image-----path",f"{image_path}_1.png")
        f_path=f"{image_path}_1.png"
        return send_from_directory(directory=os.path.join(os.getcwd(),"cropped_images"), path=f_path)

    else:
        return send_from_directory(directory=os.path.join(os.getcwd(),"UploadedFiles"), path=filename.split("\\")[-1])


@app.route("/pdfimages/<path:filename>")
def get_pdf_image(filename):
    print("----------------",filename)
    return send_from_directory(directory=os.path.join(os.getcwd(),"cropped_images"), path=filename.split("\\")[-1])


def print_prcesses():
    for thread in ALL_THREADS:
        print(thread)


def eval_data(id,comments):
    print(id,comments)
    doc_file = get_file_by_id(id)
    file_path = doc_file['file_path']
    print("--------------file path is -----------",file_path)

    cf = ClaudeEvaluator(
        image_url=file_path,
        doc_id=id,
        prompt_instructions=comments,
        formatted_data=doc_file.get("formatted_data")
    )
    print("fomatted data is ------------",cf.formatted_data)
    formatted=cf.start_eval()
    print("*" * 20, type(formatted))
    replace_none_with_empty_string(formatted)
    formatted = replace_single_quotes(formatted)
    try:
        if isinstance(formatted, dict):
            pass
        else:
            formatted = json.loads(formatted)
        print("formated is ----------", type(formatted))
    except json.JSONDecodeError as e:
        parser=SimpleJsonOutputParser()
        parsed_json=parser.parse(formatted)
        if isinstance(parsed_json,(dict,list)):
            formatted=parsed_json
        else:
            formatted=None
        print("-----------     ERROR  -----", e)
    if formatted is not None:
        replace_none_with_empty_string(formatted)
        formatted = replace_single_quotes(formatted)
        formated_2=doc_file.get("formatted_data2")
        if formated_2:
            data = {"formatted_data3": formatted}
        else:
            data = {"formatted_data2": formatted}
        add_extracted_formatted_data(id, data)
        socketio.emit(
            890 - "update",
            {
                "message": "Background task completed!",
                "result": data,
                "formatted_data2": formatted,
                "status": True,
            },
        )
        ALL_THREADS.pop(0)
    else:
        formated_2=doc_file.get("formatted_data2")
        if formated_2:
            data = {"formatted_data3": "Error formatting data"}
        else:
            data = {"formatted_data2": "Error formatting data"}

        add_extracted_formatted_data(id, data)
        socketio.emit(
            "update",
            {
                "message": "Background task could not be completed!",
                "result": data,
                "formatted": formatted,
                "status": False,
            },
        )
        ALL_THREADS.pop(0)
        print("completed------------------------------------------------------")


def print_processes():
    print("printing processes ----------",len(ALL_THREADS))
    for thread in ALL_THREADS:
        print(thread)


@app.route("/evaluate/<id>", methods=["POST"])
def evaluate_claude(id):
    print("                 inside the evaluete caide 2         ---------------")
    # doc_file = get_file_by_id(id)
    print("----------",id) 
    comments=request.json.get("comments")
    import time
    time.sleep(2)
    thread = threading.Thread(target=eval_data, args=(id,comments))
    thread.start()
    ALL_THREADS.append(thread)
    print_processes()
    return jsonify({"response":"Ok"})

if __name__ == "__main__":
    if not os.path.exists('UploadedFiles'):
        os.makedirs('UploadedFiles')
    socketio.run(app,host="0.0.0.0",port=5001 ,debug=True)