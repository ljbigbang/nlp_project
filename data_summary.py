import os
import json
from openai import OpenAI
#researcher_label
#personal background, research interest, publication, recruitment

os.environ["NVIDIA_API_KEY"] = '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'
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key='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'
)
def loadjson_categorize_researchercontent(folder_path):
    documents=[]
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path=os.path.join(folder_path,filename)
            with open(file_path,'r',encoding='utf-8') as file:
                data=json.load(file)
                researcher_content_classification = {
                    'name': 'name',
                    'personal background': [],
                    'research interest': [],
                    'publication': [],
                    'recruitment': [],
                }
                if isinstance(data,list):
                    for item in data:
                        if isinstance(item,dict) and 'name' in item and 'meta_data' in item and 'content' in item:
                            researcher_content_classification['name']=item['name']
                            if 'personal background' or 'education background' in item['meta_data']:
                                researcher_content_classification['personal background'].append(item['content'])
                            if 'research interest' in item['meta_data']:
                                researcher_content_classification['research interest'].append(item['content'])
                            if 'publication' in item['meta_data']:
                                researcher_content_classification['publication'].append(item['content'])
                            if 'recruitment' or 'lab' or 'awards' in item['meta_data']:
                                researcher_content_classification['recruitment'].append(item['content'])
                    documents.append(researcher_content_classification)
    return documents
# Concatenate the content lists of each category for each researcher into a string
def format_researchercontent(researcher_content:json):
    researcher_content['personal background']="".join(item for item in researcher_content['personal background'])
    researcher_content['research interest'] = "".join(item for item in researcher_content['research interest'])
    researcher_content['publication'] = "".join(item for item in researcher_content['publication'])
    researcher_content['recruitment'] = "".join(item for item in researcher_content['recruitment'])
    return researcher_content
def split_text(text, max_length):
    """Divide the text into blocks, with each block not exceeding the maximum length character"""
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        # Check if adding the current word exceeds the maximum length
        if len(' '.join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            # Add the current block to the result and start a new block
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
    # Add the last block
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
def summarize_longtext(category,text,max_length=3000):
    chunks = split_text(text, max_length)
    summaries = []
    for text in chunks:
        if category=='personal background':
            prompt = f"""
            give the text: {text} \n
            instruction: Please extract the content about researcher's background of the text. If you can't find relevent content, please don't output. \n
            output format: strictly answer in this format: ***** content: *****
            """
        if category=='research interest':
            prompt = f"""
            give the text: {text} \n
            instruction: Please extract the content about researcher's research interest of the text. If you can't find relevent content, please don't output. \n
            output format: strictly answer in this format: ***** content: *****
            """
        if category=='publication':
            prompt = f"""
            give the text: {text} \n
            instruction: Please extract the content of the text including the title, published time, conference of researcher's publications. If you can't find relevent content, please don't output. \n
            output format: strictly answer in this format: ***** content: *****
            """
        if category=='recruitment':
            prompt = f"""
            give the text: {text} \n
            instruction: Please extract the content about researcher's recruitment information, salary and lab condition of the text. If you can't find relevent content, please don't output. \n
            output format: strictly answer in this format: ***** content: *****
            """
        completion = client.chat.completions.create(
            model="meta/llama-3.2-3b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            top_p=1,
            max_tokens=500,
            stream=False
        )
        summary = completion.choices[0].message.content
        print(summary)
        summaries.append(summary)
    final_summary = ' '.join(summaries)
    return final_summary
def llm_content_summary(docs:list):
    for doc in docs:
        researcher_content = format_researchercontent(doc)
        background_text = researcher_content['personal background']
        interest_text = researcher_content['research interest']
        publication_text = researcher_content['publication']
        recruitment_text = researcher_content['recruitment']

        researcher_content['personal background'] = summarize_longtext('personal background',background_text)
        print("personal background finished!")
        researcher_content['research interest']=summarize_longtext('research interest',interest_text)
        print("research interest finished!")
        researcher_content['publication']=summarize_longtext('publication',publication_text)
        print("publication finished!")
        researcher_content['recruitment']=summarize_longtext('recruitment',recruitment_text)
        print("recruitment finished!")

        name=researcher_content['name']
        output_file=f'./researcher_summary_json/{name}.json'
        # used for manual evaluation
        ## output_file=f'./test_summary_json/{name}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(researcher_content, f, ensure_ascii=False, indent=4)
        print(f"The data has been successfully written {output_file}")

docs=loadjson_categorize_researchercontent('./researcher_json')
# used for manual evaluation
## docs=loadjson_categorize_researchercontent('./test_json')
llm_content_summary(docs)