import lancedb
from utils import load_json_file
from MultimodalEmbeddings import MultimodalEmbeddings
from multimodal_lancedb import MultimodalLanceDB
import pandas as pd
# declare host file
LANCEDB_HOST_FILE = "lancedb"
# declare table name
TBL_NAME = "test_tbl_1"
# initialize vectorstore
db = lancedb.connect(LANCEDB_HOST_FILE)

# load metadata files
vid1_metadata_path = 'Data/metadatas.json'

vid1_metadata = load_json_file(vid1_metadata_path)


# collect transcripts and image paths
vid1_trans = [vid['transcript'] for vid in vid1_metadata]
vid1_img_path = [vid['extracted_frame_path'] for vid in vid1_metadata]

# for video1, we pick n = 7
n = 6
updated_vid1_trans = [
 ' '.join(vid1_trans[i-int(n/2) : i+int(n/2)]) if i-int(n/2) >= 0 else
 ' '.join(vid1_trans[0 : i + int(n/2)]) for i in range(len(vid1_trans))
]

# also need to update the updated transcripts in metadata
for i in range(len(updated_vid1_trans)):
    vid1_metadata[i]['transcript'] = updated_vid1_trans[i]

print(f'A transcript example before update:\n"{vid1_trans[14]}"')
print()
print(f'After update:\n"{updated_vid1_trans[14]}"')

embedder = MultimodalEmbeddings()

_ = MultimodalLanceDB.from_text_image_pairs(texts=updated_vid1_trans,
                                           image_paths=vid1_img_path,
                                           embedding=embedder,
                                           metadatas=vid1_metadata,
                                            connection=db,
                                            table_name=TBL_NAME,
                                            mode="overwrite", )

tbl = db.open_table(TBL_NAME)

print(f'There are {tbl.to_pandas().shape[0]} rows in the table')
tbl.to_pandas()[['text', 'image_path']].head(5)