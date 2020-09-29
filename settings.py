#RUTA_ABS = '\\Users\miller.garcia\Documents\interedes\server'
RUTA_ABS = '/code/'
#WS_SCHEMA = 'wss'
WS_SCHEMA = 'ws'
WS_URL = '192.168.200.38:8000/ws/aiortc/'
#WS_URL = 'viclass.co/ws/aiortc/'
MAX_IMGS_COMPARE = 5
MAX_IMGS = 1
ALTO_RECT = 160
ANCHO_RECT = 160
#PATH_TO_CKPT = '/Users/miller.garcia/Documents/interedes/server/model/frozen_inference_graph_face.pb'
#CATEGORY_INDEX = {2:{ 'id': 2 , 'name': 'background'},1:{ 'id': 2 , 'name': ''}}
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'
NUM_CLASSES = 2
DIR_RESULTS = './rostrosGuardados'
