###########################################################
## Alunos : Keith Tsukada Sasaki            9293414      ## 
##          Gabriel Francischini de Souza   9361052      ##
###########################################################

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
from PIL import Image

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
altura = 1600
largura = 1200
window = glfw.create_window(largura, altura, "Malhas e Texturas", None, None)
glfw.make_context_current(window)

vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        varying vec2 out_texture;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
        }
        """

fragment_code = """
        uniform vec4 color;
        varying vec2 out_texture;
        uniform sampler2D samplerTexture;
        
        void main(){
            vec4 texture = texture2D(samplerTexture, out_texture);
            gl_FragColor = texture;
        }
        """       

# Request a program and shader slots from GPU
program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)

# Set shaders source
glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)

# Compile shaders
glCompileShader(vertex)
if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Vertex Shader")

glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Fragment Shader")

# Attach shader objects to the program
glAttachShader(program, vertex)
glAttachShader(program, fragment)


# Build program
glLinkProgram(program)
if not glGetProgramiv(program, GL_LINK_STATUS):
    print(glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')

# Make program the default program
glUseProgram(program)

def load_model_from_file(filename):
    """Loads a Wavefront OBJ file. """
    objects = {}
    vertices = []
    texture_coords = []
    faces = []

    material = None

    # abre o arquivo obj para leitura
    for line in open(filename, "r"): ## para cada linha do arquivo .obj
        if line.startswith('#'): continue ## ignora comentarios
        values = line.split() # quebra a linha por espaço
        if not values: continue

        ### recuperando vertices
        if values[0] == 'v':
            vertices.append(values[1:4])


        ### recuperando coordenadas de textura
        elif values[0] == 'vt':
            texture_coords.append(values[1:3])

        ### recuperando faces 
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'f':
            face = []
            face_texture = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                if len(w) >= 2 and len(w[1]) > 0:
                    face_texture.append(int(w[1]))
                else:
                    face_texture.append(0)

            faces.append((face, face_texture, material))

    model = {}
    model['vertices'] = vertices
    model['texture'] = texture_coords
    model['faces'] = faces

    return model

glEnable(GL_TEXTURE_2D)
qtd_texturas = 30
textures = glGenTextures(qtd_texturas)

def load_texture_from_file(texture_id, img_textura):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    img = Image.open(img_textura)
    img_width = img.size[0]
    img_height = img.size[1]
    image_data = img.tobytes("raw", "RGB", 0, -1)
    #image_data = np.array(list(img.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)

vertices_list = []    
textures_coord_list = []

def processa_modelo(modelo, arquivo):
    print(f'Processando modelo {arquivo}.obj. Vertice inicial = {len(vertices_list)}')
    for face in modelo['faces']:
        for vertice_id in face[0]:
            vertices_list.append( modelo['vertices'][vertice_id-1] )
        for texture_id in face[1]:
         textures_coord_list.append( modelo['texture'][texture_id-1] )
    print(f'Processando modelo {arquivo}.obj. Vertice final = {len(vertices_list)}')

def processa_modelo_multipla_textura(modelo, arquivo):
    print(f'Processando modelo {arquivo}.obj. Vertice inicial = {len(vertices_list)}')
    faces_visited = []
    for face in modelo['faces']:
        if face[2] not in faces_visited:
            print(f'{face[2]} vertice inicial = {len(vertices_list)}')
            faces_visited.append(face[2])
        for vertice_id in face[0]:
           vertices_list.append( modelo['vertices'][vertice_id-1] )
        for texture_id in face[1]:
            textures_coord_list.append( modelo['texture'][texture_id-1] )
    print(f'Processando modelo {arquivo}.obj. Vertice final = {len(vertices_list)}')

## GRAMA - não está sendo desenhado
modelo = load_model_from_file('terreno/grama.obj')
processa_modelo(modelo,"grama")
## PISO DA CASA
modelo = load_model_from_file('terreno/piso.obj')
processa_modelo(modelo,'piso')
## ASFALTO
modelo = load_model_from_file('terreno/asfalto.obj')
processa_modelo(modelo,'asfalto')
## CASA
modelo = load_model_from_file('casa/casa.obj')
processa_modelo_multipla_textura(modelo,'casa')
## PLANTA1
modelo = load_model_from_file('plantas/plantas.obj')
processa_modelo_multipla_textura(modelo,'plantas')
## POKEMON INTERNO
modelo = load_model_from_file('animal/pokemon.obj')
processa_modelo(modelo,'pokemon')
## BANCO EXTERNO
modelo = load_model_from_file('objeto/banco.obj')
processa_modelo_multipla_textura(modelo,'banco')
## POKEMON EXTERNO
modelo = load_model_from_file('animal/pokemon2.obj')
processa_modelo(modelo,'pokemon2')
#CADEIRA1
modelo = load_model_from_file('objeto/cadeira.obj')
processa_modelo(modelo,'cadeira')
#CADEIRA2
modelo = load_model_from_file('objeto/cadeira2.obj')
processa_modelo_multipla_textura(modelo,'cadeira2')

modelo = load_model_from_file('ceu/ceu.obj')
processa_modelo(modelo,'ceu')
## CARREGANDO TEXTURAS
load_texture_from_file(0,'terreno/grama.jpg')
load_texture_from_file(1,'terreno/piso.jpg')
load_texture_from_file(2,'terreno/asfalto.jpg')
load_texture_from_file(3,'casa/tex/1.png')
load_texture_from_file(4,'casa/tex/floor.png')
load_texture_from_file(5,'casa/tex/roof and pillars.png')
load_texture_from_file(6,'casa/tex/porta.png')
load_texture_from_file(7,'casa/tex/door front and back.png')
load_texture_from_file(8,'casa/tex/windows & stairs.png')
load_texture_from_file(9,'casa/tex/grass.png')
load_texture_from_file(10,'plantas/DracenaMarginataLEaf_red.jpg')
load_texture_from_file(11,'plantas/dracenaMarginata_bark1.jpg')
load_texture_from_file(12,'plantas/dracenaMarginata_bark2.jpg')
load_texture_from_file(13,'plantas/MUG.jpg')
load_texture_from_file(14,'plantas/pebbleTex.jpg')
load_texture_from_file(15,'plantas/concreteWhite.jpg')
load_texture_from_file(16,'animal/pokemon.png')
load_texture_from_file(17,'objeto/woods.png')
load_texture_from_file(18,'objeto/steel.png')
load_texture_from_file(19,'animal/pokemon2.png')
load_texture_from_file(20,'objeto/cadeira.png')
load_texture_from_file(21,'objeto/cadeira2_suporte.jpg')
load_texture_from_file(22,'objeto/cadeira2.jpg')
load_texture_from_file(23,'ceu/estrelado.jpg')

# Request a buffer slot from GPU
buffer = glGenBuffers(2)

vertices = np.zeros(len(vertices_list), [("position", np.float32, 3)])
vertices['position'] = vertices_list

# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[0])
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
stride = vertices.strides[0]
offset = ctypes.c_void_p(0)
loc_vertices = glGetAttribLocation(program, "position")
glEnableVertexAttribArray(loc_vertices)
glVertexAttribPointer(loc_vertices, 3, GL_FLOAT, False, stride, offset)

textures = np.zeros(len(textures_coord_list), [("position", np.float32, 2)]) # duas coordenadas
textures['position'] = textures_coord_list

# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[1])
glBufferData(GL_ARRAY_BUFFER, textures.nbytes, textures, GL_STATIC_DRAW)
stride = textures.strides[0]
offset = ctypes.c_void_p(0)
loc_texture_coord = glGetAttribLocation(program, "texture_coord")
glEnableVertexAttribArray(loc_texture_coord)
glVertexAttribPointer(loc_texture_coord, 2, GL_FLOAT, False, stride, offset)

#Classe que armazena as informações de um objeto de textura para ser utilizada na funcao de desenhar
class Textura:
  def __init__(self, inicio, fim,id):
    self.inicio = inicio
    self.fim = fim
    self.id = id

#funcao que recebe os valores necessários desenhando-o na tela
def desenha(ang,r_x,r_y,r_z,t_x,t_y,t_z,s_x,s_y,s_z, texturas):
     mat_model = model(ang, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
     loc_model = glGetUniformLocation(program, "model")
     glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)

     for k in range(len(texturas)):
         glBindTexture(GL_TEXTURE_2D,texturas[k].id)
         glDrawArrays(GL_TRIANGLES, texturas[k].inicio, texturas[k].fim - texturas[k].inicio)

def limita_bordas(pos, front, limites, key):

    if (pos >= limites[0] and pos <= limites[1]):
        return True
    elif (pos < limites[0] and front > 0) or (pos > limites[1] and front < 0):
        if key == 87:
            return True
    elif (pos < limites[0] and front < 0) or (pos > limites[1] and front > 0):
        if key == 83:
            return True

    return False 

cameraPos   = glm.vec3(0.0,  0.0,  1.0);
cameraFront = glm.vec3(0.0,  0.0, -1.0);
cameraUp    = glm.vec3(0.0,  1.0,  0.0);
polygonal_mode = False

def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp, polygonal_mode

    lim_x = limita_bordas(cameraPos.x,cameraFront.x, [-2.0, 27.0], key)
    lim_y = limita_bordas(cameraPos.y,cameraFront.y, [0.0, 28.0], key)
    lim_z = limita_bordas(cameraPos.z,cameraFront.z, [-14.0, 18.0], key)

    if(lim_x and lim_y and lim_z):
        cameraSpeed = 0.5
    else:
        cameraSpeed = 0.0

    if key == 87 and (action==1 or action==2): # tecla W
        cameraPos += cameraSpeed * cameraFront
    
    if key == 83 and (action==1 or action==2): # tecla S
        cameraPos -= cameraSpeed * cameraFront
    
    if key == 65 and (action==1 or action==2): # tecla A
        cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 68 and (action==1 or action==2): # tecla D
        cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 80 and action==1 and polygonal_mode==True:
        polygonal_mode=False
    else:
        if key == 80 and action==1 and polygonal_mode==False:
            polygonal_mode=True
        
        
        
firstMouse = True
yaw = -90.0 
pitch = 0.0
lastX =  largura/2
lastY =  altura/2

def mouse_event(window, xpos, ypos):
    global firstMouse, cameraFront, yaw, pitch, lastX, lastY
    if firstMouse:
        lastX = xpos
        lastY = ypos
        firstMouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

    sensitivity = 0.3 
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset;
    pitch += yoffset;

    
    if pitch >= 90.0: pitch = 90.0
    if pitch <= -90.0: pitch = -90.0

    front = glm.vec3()
    front.x = math.cos(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    front.y = math.sin(glm.radians(pitch))
    front.z = math.sin(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    cameraFront = glm.normalize(front)


    
glfw.set_key_callback(window,key_event)
glfw.set_cursor_pos_callback(window, mouse_event)

def model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z):
    
    angle = math.radians(angle)
    
    matrix_transform = glm.mat4(1.0) # instanciando uma matriz identidade
       
    # aplicando rotacao
    matrix_transform = glm.rotate(matrix_transform, angle, glm.vec3(r_x, r_y, r_z))
        
  
    # aplicando translacao
    matrix_transform = glm.translate(matrix_transform, glm.vec3(t_x, t_y, t_z))    
    
    # aplicando escala
    matrix_transform = glm.scale(matrix_transform, glm.vec3(s_x, s_y, s_z))
    
    matrix_transform = np.array(matrix_transform).T # pegando a transposta da matriz (glm trabalha com ela invertida)
    
    return matrix_transform

def view():
    global cameraPos, cameraFront, cameraUp
    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    mat_view = np.array(mat_view)
    return mat_view

def projection():
    global altura, largura
    # perspective parameters: fovy, aspect, near, far
    mat_projection = glm.perspective(glm.radians(45.0), largura/altura, 0.1, 1000.0)
    mat_projection = np.array(mat_projection)    
    return mat_projection


glfw.show_window(window)
glfw.set_cursor_pos(window, lastX, lastY)

glEnable(GL_DEPTH_TEST) ### importante para 3D
   

t_z_pokemon2 = 5.0
t_y_pokemon2 = -0.8
y = 0.05
while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    if polygonal_mode==True:
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    if polygonal_mode==False:
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
    
    t = []
    # PISO
    t.append(Textura(6,12,1))
    desenha(0.0,0.0,0.0,1.0,12.5,-1.01,0.0,15.0,1.0,15.0,t)
    t.clear()
    # ASFALTO
    t.append(Textura(12,18,2))
    desenha(90.0,0.0,90.0,0.0,-17.5,-1.01,12.5,2.5,1.0,15.0,t)
    t.clear()
    # CASA
    t = [ Textura(18,1365,3), Textura(1395,1461,4), Textura(1461,5415,5), Textura(5415,5787,6), Textura(5787,10443,7),
    Textura(10443,12003,8), Textura(12003,12426,9)]
    desenha(0.0,0.0,0.0,1.0,12.5,-1.0,0.0,0.05,0.05,0.05,t)
    t.clear()
    # PLANTAS
    t = [Textura(12426,2511306,10), Textura(2511306,2545914,11), Textura(2545914,2644626,12),
    Textura(2644626,2672256,13), Textura(2672256,3761256,14), Textura(3761256,3855336,15)]
    desenha(0.0,0.0,0.0,1.0,10.0,-0.4,5.0,0.2,0.2,0.2,t)
    #POKEMON 1
    t = [Textura(3855336,7125864,16)]
    desenha(0.0,0.0,0.0,1.0,15.0,-0.21,-3.0,0.2,0.2,0.2,t)
    #BANCO
    t = [Textura(7125864,7126224,17), Textura(7126224,7510128,18)]
    desenha(0.0,0.0,1.0,0.0,13.0,-0.8,6.0,0.02,0.02,0.02,t)
    #POKEMON 2
    t = [Textura(7510128,12693360,19)]
    desenha(0.0,0.0,0.0,1.0,18.0,t_y_pokemon2,t_z_pokemon2,0.02,0.02,0.02,t)
    t_z_pokemon2 += 0.02
    if(t_z_pokemon2 >= 19):
        t_z_pokemon2 = 5.0

    t_y_pokemon2 += y
    if(t_y_pokemon2 >= 0.5 or t_y_pokemon2 <= -0.8 ):
        y *= -1

    #CADEIRA 1
    t = [Textura(12693360,12709800,20)]
    desenha(0.0,0.0,0.0,1.0,18.0,-0.36,-3.0,0.2,0.2,0.2,t)
    #CADEIRA 2
    t = [Textura(12709800,12744936,21),Textura(12744936,12769128,22)]
    desenha(0.0,0.0,1.0,1.0,12.0,-0.36,-3.0,0.4,0.4,0.4,t)


    ##desenha(ang,r_x,r_y,r_z,t_x,t_y,t_z,s_x,s_y,s_z, texturas):
    #CEU
    t = [Textura(12769128,12769134,23)]
    desenha(90,0.0,0.0,1.0,14.0,2.5,2.5,15.0,15.0,17.5,t)#esquerda 
    desenha(90,0.0,0.0,1.0,14.0,-27.5,2.5,15.0,15.0,17.5,t)#direita
    desenha(90,1.0,0.0,0.0,12.5,-15.0,-14.0,15.0,15.0,15.0,t)#frente
    desenha(90,1.0,0.0,0.0,12.5,20.0,-14.0,15.0,15.0,15.0,t)#atras
    desenha(0.0,0.0,0.0,1.0,12.5,29.0,2.5,15.0,15.0,17.5,t)#cima

    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_FALSE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_FALSE, mat_projection)    
    
    glfw.swap_buffers(window)

glfw.terminate()