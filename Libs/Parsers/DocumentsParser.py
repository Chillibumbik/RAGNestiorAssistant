"""
Здесь прописан парсинг PDF, TXT, Docx, Markdown, PPTx
"""

from langchain_community.document_loaders import (UnstructuredFileLoader, UnstructuredMarkdownLoader,
                                                  UnstructuredPowerPointLoader,
                                                  UnstructuredWordDocumentLoader, UnstructuredPDFLoader)

from pathlib import Path
from langchain.schema import Document
from tqdm import tqdm

def get_document_loader(document_path: str | Path, mode: str = 'single',
                        include_page_breaks: bool = False) -> UnstructuredFileLoader | Exception:
    """
    document_path - путь к документу
    mode - режим парсинга
    include_page_breaks - включает ли склейку страниц

    Возвращает парсер документа
    """

    document_path = Path(document_path)

    match document_path.suffix:
        case ".pdf":
            return UnstructuredPDFLoader(document_path.as_posix(),
                                         mode=mode,  # чтобы возвращалось как один объект
                                         strategy='auto',  # чтобы и сканы и обычные пдф читать
                                         languages=['rus', 'eng'],
                                         include_page_breaks=include_page_breaks)  # склейка страниц
        case ".docx":
            return UnstructuredWordDocumentLoader(document_path.as_posix(),
                                                  mode=mode,
                                                  strategy='auto',
                                                  languages=['rus', 'eng'],
                                                  include_page_breaks=include_page_breaks)
        case ".md":
            return UnstructuredMarkdownLoader(document_path.as_posix(),
                                              mode=mode,
                                              strategy='auto',
                                              languages=['rus', 'eng'],
                                              include_page_breaks=include_page_breaks)
        case ".pptx":
            return UnstructuredPowerPointLoader(document_path.as_posix(),
                                                mode=mode,
                                                strategy='auto',
                                                languages=['rus', 'eng'],
                                                include_page_breaks=include_page_breaks)
        case _:
            return Exception(f"Неподдерживаемый формат файла: {document_path.suffix}")


def parse_document(document_path: str | Path, **kwargs) -> list[Document] | Exception:
    """
    document_path - путь к документу

    Возвращает список документов
    """
    try:
        document_path = Path(document_path, **kwargs)

        loader = get_document_loader(document_path)

        for doc in loader.load():
            doc.metadata["source"] = document_path.name
            doc.metadata["source_path"] = str(document_path)
            doc.metadata["type"] = str(doc.__class__.__name__)

        return loader.load()

    except Exception as e:
        print(f'{document_path} ParseError: {e}')
        return e


def parse_documents_in_dir(
        dir_path: str | Path,
        recursive: bool = True,
        **kwargs,
) -> list[Document] | Exception:
    """
    Парсинг файлов в указанной директории и всех вложенных директориях.

    Args:
        dir_path (str | Path): Путь к директории с файлами.
        recursive (bool): Производить ли рекурсивный поиск вложенных директорий.
        **kwargs: Дополнительные аргументы для функции парсинга.

    Returns:
        list[Document] | Exception: Список объектов Document, если найдены файлы и успешно распарсены, иначе Exception.
    """
    dir_path = Path(dir_path)

    out: list[Document] = []

    for doc in tqdm(dir_path.glob('**/*' if recursive else '*')):  # Рекурсивно ищем во всех вложенных директориях
        docs = parse_document(doc, **kwargs)
        if docs:
            out.extend(docs)

    # Возвращаем список документов, если он не пустой, иначе сообщение об ошибке
    return out if out else Exception(f"Неверный путь: {dir_path}")

# тест
if __name__ == "__main__":
    res = parse_document(r"D:\pythonProject\Codes\Folder for study\ML_study\third_practical_task\Документы НИЯУ МИФИ\Изменения в Устав НИЯУ МИФИ от 05.02.2025.pdf")
    print(res)