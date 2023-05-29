import os
import re
import PyPDF2
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from config.config import TOKEN

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = TOKEN

# Open the PDF file
with open('../input/ABC123_verificacion_vehicular.pdf', 'rb') as pdfFileObj:
    # Create a PDF reader object
    pdfReader = PyPDF2.PdfReader(pdfFileObj)

    # Get the number of pages in the PDF file
    print(f"Number of pages: {len(pdfReader.pages)}")

    # Get the first page of the PDF
    pageObj = pdfReader.pages[0]

    # Extract the text from the page and remove extra spaces between characters
    text = re.sub(r"(?<=\w) (?=\w)", "", pageObj.extract_text())

    # Search for the license plate in the text
    match = re.search(r'(placa|Número de Placa|patente|license_plate)(.{0,30})', text, re.IGNORECASE | re.DOTALL)
    if match:
        segment = match.group(0)  # The license plate and the following 30 characters
    else:
        print("No license plate found in the document.")
        exit(1)

# Construct the content for the chat
content = "You are a text interpreter API. Your responses should always be in JSON format, using the following " \
          "structure: {\"result\": \"$result\"}. Now, please search the license_plate or also called placa in spanish in " \
          "the following text: " + segment

# Initialize the chat model
chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Use the OpenAI callback to print the chat response and callback
with get_openai_callback() as cb:
    print(chat([HumanMessage(content=content)]))
    print(cb)
