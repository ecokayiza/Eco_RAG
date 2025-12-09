from config import Config
from utils import Assembler

file_path = Config.TEST_FILE_PATH

print("Dpocument count in DB before storing file:", Assembler.db.count())
Assembler.store_file(file_path)
print("Document count in DB after storing file:", Assembler.db.count())

# print("Document count in DB before deleting file:", Assembler.db.count())
# Assembler.delete_file(file_path)
# print("Document count in DB after deleting file:", Assembler.db.count())

# results = Assembler.query_file(file_path)


# text = ["智能体采取的动作"]
# embedding = HuggingFaceEmbedder.embed(text)[0]
# results = Assembler.query_with_vector(embedding, n_results=1)
# Assembler.results_formatted_viewer(results)