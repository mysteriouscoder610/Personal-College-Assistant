import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

brief_prompt = PromptTemplate(
    template = '''Create a BRIEF summary with this format:
        **MAIN TOPIC**: [One sentence]
        **KEY POINTS**: [3-4 bullet points]
        **CONCLUSION**: [One sentence takeaway]

        Requirements:
        - Be 100 percent accurate to the source
        - Keep it very concise (under 150 words)
        - Focus on the most essential information
        - {text}
        ''',
        input_variables = ['text']
    )

detailed_prompt = PromptTemplate(
    template = '''Create a DETAILED summary with this format:
        **MAIN TOPIC**: [One sentence]
        **KEY POINTS**: [5-7 bullet points]
        **CRITICAL INSIGHTS**: [Most important discoveries]
        **METHODOLOGY**: [How the research was conducted]
        **FINDINGS**: [Key results and data]
        **CONCLUSION**: [Final takeaway and implications]

        Requirements:
        - Be 100 percent accurate to the source
        - Use engaging formatting
        - Highlight what matters most
        - Make it visually appealing
        - Be comprehensive and thorough
        - {text}
        ''',
        input_variables = ['text']
    )

parser = StrOutputParser()

loader = PyPDFLoader("")

docs = loader.load()

# User choice for summary type
print("Choose summary type:")
print("1. Brief summary")
print("2. Detailed summary")
choice = input("Enter your choice (1 or 2): ")

if choice == "1":
    chain = brief_prompt | model | parser
    print("\n=== BRIEF SUMMARY ===")
elif choice == "2":
    chain = detailed_prompt | model | parser
    print("\n=== DETAILED SUMMARY ===")
else:
    print("Invalid choice. Using brief summary by default.")
    chain = brief_prompt | model | parser
    print("\n=== BRIEF SUMMARY ===")

print(chain.invoke({"text": docs[0].page_content}))



