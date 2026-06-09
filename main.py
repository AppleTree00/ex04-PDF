# pip install -U langchain-community pypdf

from langchain_community.document_loaders import PyPDFLoader

file_path = "unsu01.pdf"

# PDF 파일에서 페이지 객체를 추출하여 리스트로 반환하는 PyPDFLoader 클래스의 인스턴스를 생성합니다.
loader = PyPDFLoader(file_path)

# PDF 파일에서 페이지를 로드하고 분할하여 페이지 객체 리스트로 반환
pages = loader.load_and_split()

if len(pages) > 1:
    print("-----[두번째 페이지 객체 전체 출력]-----")
    print(pages[1])

    print("-----[두번째 페이지 실제 텍스트 객체 전체 출력]-----")
    print(pages[1].page_content)
    print("그냥 페이지 객체를 출력하면 페이지 객체의 메타데이터와 텍스트가 함께 출력됩니다.")
else:
    print(f"PDF 파일에 페이지가 하나도 없습니다: {file_path}")