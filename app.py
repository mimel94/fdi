import argparse
import asyncio
import json
import logging
import os
import fnmatch
import ssl
import uuid
import aiohttp
import cv2
import numpy as np
from scipy.misc import imread
import multiprocessing
import threading
import time
import tensorflow as tf
from werkzeug.utils import secure_filename
from lib.src.align import detect_face  # for MTCNN face detection
from utils import (
    load_model,
    get_face,
    get_faces_live,
    forward_pass,
    save_embedding,
    load_embeddings,
    identify_face,
    allowed_file,
    remove_file_extension,
    save_image
)
from threading import Timer
from aiohttp import web
from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from settings import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger("pc")
pcs = set()

usuario_haar_cargado = False
usuario_haar_creando = False
comparando_usuario = False
comparando_usuario_resultado = False
comparando_usuario_parametro = 0
intervalo_tomar_foto = 0
validar_usuario = True   
flag = ''
tomar_foto= True 
cont = 0
cont2 = 0
cont3 = 0
mensaje_socket = ''
mensaje_error = ''
rects = []
make_rects = True

# Load FaceNet model and configure placeholders for forward pass into the FaceNet model to calculate embeddings
model_path = 'model/20170512-110547/20170512-110547.pb'
facenet_model = load_model(model_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
image_size = 160
image2Train = ''
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

# Initiate persistent FaceNet model in memory
facenet_persistent_session = tf.Session(graph=facenet_model, config=config)

# Create Multi-Task Cascading Convolutional (MTCNN) neural networks for Face Detection
pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)


class VideoTransmision(MediaStreamTrack):
    """
    Fuente de video para Reconocimiento facial viclass
    """
    kind = "video"
    def __init__(self, track, accion, usuario, socket, cuenta, rutaImg, rutaHaar):
        super().__init__()  # don't forget this!
        self.track = track
        self.accion = accion
        self.usuario = usuario
        self.socket = socket
        self.rutaImg = rutaImg
        self.rutaHaar = rutaHaar
        self.cuenta = cuenta      

    async def recv(self):
        frame = await self.track.recv()
        global validar_usuario
        global cont
        global image2Train
        global embeddings
        global mensaje_socket
        global cont2
        global flag
        global rects
        global make_rects

        image = frame.to_ndarray(format="bgr24")        
        if self.accion == 'entrenar' :      
            if self.accion == flag:                
                flag=''
                #cv2.imwrite(self.rutaImg +'/'+str(self.usuario)+'.jpg'.format(self.cuenta), image)
                #self.cuenta += 1                    
                #image2Train = imread(self.rutaImg+'/'+str(self.usuario)+'.jpg', mode='RGB' )                
                filename = str(self.usuario)+'.jpg'   
                crearEmbeding = threading.Thread(target=crearEmbeddingUsuario,args=(image, filename, self.rutaHaar,self.socket,))             
                crearEmbeding.start()                   
            if mensaje_socket:                    
                await self.socket.send_str(mensaje_socket)  
                mensaje_socket=''                   
        elif self.accion == 'comparar' :             
            if self.accion == flag:     
                flag=''           
                embedding_dict = load_embeddings(self.usuario)                 
                validarUsuario = threading.Thread(target=validarUsuarioFacenet, args=(embedding_dict,image,))
                validarUsuario.start()                                                  
            if mensaje_socket:               
                await self.socket.send_str(mensaje_socket)  
                mensaje_socket = ''  
        if make_rects:
            make_rects = False
            _, rects = get_faces_live(
                    img =image,
                    pnet=pnet,
                    rnet=rnet,
                    onet=onet,
                    image_size=image_size
                )
            thread_rects = Timer(0.4,generate_rects)
            thread_rects.start()
        if rects:
            for i in range(len(rects)):
                rect = rects[i]
                rect = [coordinate * 1 for coordinate in rect]
                cv2.rectangle(
                    img = image, 
                    pt1=(rect[0], rect[1]),
                    pt2=(rect[2], rect[3]),
                    color=(85, 222, 172),
                    thickness=2
                )
        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
def generate_rects():
    global make_rects
    make_rects = True
def flagComparar(accion):
    global flag    
    flag = accion
    
def contador():
    global cont
    cont+=1
def resetContador():
    global cont
    cont = 0

def crearEmbeddingUsuario(image2Train, filename, rutaHaar, socket):
    global mensaje_socket
    global flag
    global cont2
    global cont  
    global cont3
    global mensaje_error   
    cambiarFlag = Timer(0.5, flagComparar,args=('entrenar',)) 
    faces,_ = get_faces_live(
        img = image2Train,
        pnet = pnet,
        rnet = rnet,
        onet = onet,
        image_size = image_size
    )           
    if len(faces) == 1 :
        embedding = forward_pass(
            img= faces[0],
            session=facenet_persistent_session,
            images_placeholder=images_placeholder,
            embeddings=embeddings,
            phase_train_placeholder=phase_train_placeholder,
            image_size=image_size
        )      
        filename = remove_file_extension(filename=filename)
        save_embedding(
            embedding=embedding,
            filename=filename,
            embeddings_path=rutaHaar
        )                
        cont3 = 1    
    elif len(faces)>1 : 
        contador()
        mensaje_error = 'multiples rostros detectados' 
        
    elif len(faces)<1 :
        contador()
        mensaje_error = 'no se detecto un rostro'  
        
    if cont == 5:
        mensaje_socket = "{'accion':'no_entrenado','razon':'"+mensaje_error+"'}"  
        mensaje_error=''
        resetContador()
    elif cont3==1:
        mensaje_socket = "{'accion':'entrenado'}"
        cont3=0
    else:
        cambiarFlag.start()
    
        
     

def validarUsuarioFacenet(embedding_dict, image):
    global flag
    global cont2
    global cont
    global mensaje_socket   
    global mensaje_error
    cambiarFlag = Timer(0.5, flagComparar,args=('comparar',)) 
    if(embedding_dict):  
        faces, _ = get_faces_live(
            img=image,
            pnet=pnet,
            rnet=rnet,
            onet=onet,
            image_size=image_size
        )            
        if len(faces)==1:
            for i in range(len(faces)):
                face_img = faces[i]                
                face_embedding = forward_pass(
                    img = face_img,
                    session= facenet_persistent_session,
                    images_placeholder=images_placeholder,
                    embeddings=embeddings,
                    phase_train_placeholder=phase_train_placeholder,
                    image_size=image_size
                )                            
                _, status = identify_face(
                    embedding=face_embedding,
                    embedding_dict=embedding_dict
                )                                            
                if status == True:
                    contador()                    
                    cont2+=1                            
                else:
                    contador()                    
        elif len(faces)>1:     
            contador()                                         
            mensaje_error = 'multiples rostros detectados'      
        elif len(faces)<1:     
            contador()                                         
            mensaje_error = 'No se detecto rostros'    
        if cont == 5:    
            if cont2 >= 4:                        
                mensaje_socket= "{'accion':'comparado','resultado':'1'}"                           
                cont2=0
                resetContador()                                                 
            else:                        
                cont2=0
                resetContador()  
                #mensaje_socket= "{'accion':'comparado','resultado':'0'}"                
                if mensaje_error:
                    mensaje_socket = "{'accion':'comparado','resultado':'0','error':'"+mensaje_error+"'}"                      
                    mensaje_error=''  
                else: 
                    mensaje_socket= "{'accion':'comparado','resultado':'0'}"                                                 
        else:                    
            cambiarFlag.start()  
async def offer(request):
    global flag
    params = await request.json()
    flag = params['accion']
    id_usuario = getId(params)
    if id_usuario < 1 :
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"error": "Sin usuario autenticado"}
            ),
        )

    rutaImg = RUTA_ABS + '/vision/imagenes/' + str(id_usuario)
    rutaHaar = RUTA_ABS + '/vision/embeddings/'+ str(id_usuario)
    
    
    if not os.path.exists(rutaImg):
        os.makedirs(rutaImg)
    if not os.path.exists(rutaHaar):
        os.makedirs(rutaHaar)
    cuenta = len(fnmatch.filter(os.listdir(rutaImg), '*.jpg'))
    #print(os.path.exists(rutaHaar +'/'+'*.npy'))
    if params["accion"] == 'detect' and not os.path.exists(rutaHaar +'/'+'*.npy'):
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"error": "Rostro no entrenado todavía"}
            ),
        )

    session = aiohttp.ClientSession()
    ws = await session.ws_connect(WS_SCHEMA + '://' + WS_URL +'?room=muestras_' + str(id_usuario))  

    if os.path.exists(rutaHaar+'/'+str(id_usuario)+'.npy'):
        #await ws.send_str("{'accion':'entrenado'}")    
        await ws.send_str("{'accion':'iniciando','status':'1'}")
    else:
        await ws.send_str("{'accion':'iniciando','status':'0'}")

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    #dataRequest = json.dumps(params)
    #print(dir(ws))
    #log_info('Parametros del request',str(request.forwarded) +';'+str(request.host)+';'+str(request.remote)+';')

    log_info("Created for %s", request.remote)
    # recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()            
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            local_video = VideoTransmision(
                track, accion=params["accion"], usuario=id_usuario, socket=ws, cuenta=cuenta, rutaImg=rutaImg, rutaHaar=rutaHaar
            )
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"id":params["id"], "sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def test(request):
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"test":"OK"}
        ),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def index(request):
    content = open(os.path.join(RUTA_ABS, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(RUTA_ABS, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


def getId(params):
    return params["id"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC para reconocimiento facial"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8085, help="Port for HTTP server (default: 8085)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/aiortc/offer", offer)
    app.router.add_get("/aiortc/test", test)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    web.run_app(app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context)

