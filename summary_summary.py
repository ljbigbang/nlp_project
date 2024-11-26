import os
import json
from openai import OpenAI
os.environ["NVIDIA_API_KEY"] = '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'
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key='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'
)
def format_researchersummary(name,summary):
    return "".join(f"name:{name} \n {summary}")
def loadjson_integrate_researchersummary(folder_path):
    researcher_summary_integration={
        'personal background': [],
        'research interest': [],
        'publication': [],
        'recruitment': []
    }
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path=os.path.join(folder_path,filename)
            with open(file_path,'r',encoding='utf-8') as file:
                data=json.load(file)
                researcher_summary_integration['personal background'].append(format_researchersummary(data['name'],data['personal background']))
                researcher_summary_integration['research interest'].append(format_researchersummary(data['name'], data['research interest']))
                researcher_summary_integration['publication'].append(format_researchersummary(data['name'], data['publication']))
                researcher_summary_integration['recruitment'].append(format_researchersummary(data['name'], data['recruitment']))
    return researcher_summary_integration
def summarize_longtext(category,chunks):
    summaries = []
    for text in chunks:
        if category=='personal background':
            prompt = f"""
            give the text: {text} \n
            instruction: Please extract the core content of the text. \n
            you must output the researcher's name first and then extract the core content of the text. \n
            output format: strictly answer in this format: ***** content: *****
            """
        if category=='research interest':
            prompt = f"""
            give the text: {text} \n
            instruction: Please extract the core content of the text. \n
            you must output the researcher's name first and then extract the core content of the text. \n
            output format: strictly answer in this format: ***** content: *****
            """
        if category=='publication':
            prompt = f"""
            give the text: {text} \n
            instruction: Please extract the core content of the text. \n
            you must output the researcher's name first and then output the title, published time, conference of researcher's publications. \n
            output format: strictly answer in this format: ***** content: *****
            """
        if category=='recruitment':
            prompt = f"""
            give the text: {text} \n
            instruction: Please extract the core content of the text. \n
            you must output the researcher's name first and then extract the core content of the text. \n
            output format: strictly answer in this format: ***** content: *****
            """
        completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
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
def llm_content_summary(doc:json):
    background_chunks = doc['personal background']
    interest_chunks = doc['research interest']
    publication_chunks = doc['publication']
    recruitment_chunks = doc['recruitment']
    doc['personal background'] = summarize_longtext('personal background', background_chunks)
    print("personal background finished!")
    doc['research interest'] = summarize_longtext('research interest', interest_chunks)
    print("research interest finished!")
    doc['publication'] = summarize_longtext('publication', publication_chunks)
    print("publication finished!")
    doc['recruitment'] = summarize_longtext('recruitment', recruitment_chunks)
    print("recruitment finished!")
    output_file = './summary_summary_json/context.json'
    # used for manual evaluation
    ## output_file='./test_summary_summary_json/context.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=4)
    print(f"The data has been successfully written {output_file}")


doc_json=loadjson_integrate_researchersummary('./researcher_summary_json')
# used for manual evaluation
## doc_json=loadjson_integrate_researchersummary('./test_summary_json')
llm_content_summary(doc_json)
