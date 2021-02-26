# [START setup]
import base64
import json
import os

from google.cloud import storage
from google.cloud import vision

vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()

project_id = None # L'ID de votre projet

with open('config.json') as f:
    data = f.read()
config = json.loads(data)
# [END setup]

# [START common functions]
def get_image_from_bucket(bucket):
    """Get image from bucket"""
    
    filename = bucket['name']
    image = vision.types.Image()
    image.source.image_uri = f"gs://{bucket['bucket']}/{filename}"

    return filename, image


def save_results(filename, result, extension='.json'):
    """Save results to bucket"""
    
    bucket_name = config['RESULT_BUCKET']
    result_filename = f'{filename}{extension}'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(result_filename)

    print(f'Saving results to {result_filename} in bucket {bucket_name}')

    blob.upload_from_string(json.dumps(result))
# [END common functions]


# [START functions_localize_objects]
def localize_objects(bucket, filename):
    """Localize objects in the image on Google Cloud Storage

    Args:
    bucket: bucket object
    filename: name of the image file
    """
   
    filename, image = get_image_from_bucket(bucket)
    objects = vision_client.object_localization(image=image).localized_object_annotations

    print(f'Number of objects found: {len(objects)}')
    for object_ in objects:
        print(f'\n{object_.name} (confidence: {object_.score})')
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(f' - ({vertex.x}, {vertex.y})')


    # Exercice
    # --------
    #
    # Construire le dictionnaire 'result' de la forme:
    #
    # result = {
    #   <id_objet_1>: {
    #       'name': <catégorie de l'object détecté>,
    #	    'score': <score de détection pour l'objet>,
    #       'vertices': [
    #            (<vertex1>), # Tuple
    #            (<vertex2>),
    #            (<vertex3>),
    #            (<vertex4>)
    #            ]
    #       },
    #   <id_objet_2>: {
    #       'name': <catégorie de l'object détecté>,
    #	    'score': <score de détection pour l'objet>,
    #       'vertices': [
    #            (<vertex1>),
    #            (<vertex2>),
    #            (<vertex3>),
    #            (<vertex4>)
    #            ]
    #       },
    #
    #   ...

    save_results(filename, result, extension='.objects.json')
# [END functions_localize_objects]


# [START functions_detect_faces]
def detect_faces(bucket, filename):
    """Detects faces in a file located in Google Cloud Storage

    Args:
    bucket: bucket object
    filename: name of the image file
    """

    filename, image = get_image_from_bucket(bucket)

    response = vision_client.face_detection(image=image)
    faces = response.face_annotations
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('Faces:')
    for face in faces:
        print(f'anger: {likelihood_name[face.anger_likelihood]}')
        print(f'joy: {likelihood_name[face.joy_likelihood]}')
        print(f'surprise: {likelihood_name[face.surprise_likelihood]}')

        vertices = ([f'({vertex.x},{vertex.y})'
                    for vertex in face.bounding_poly.vertices])

        print('face bounds: {}'.format(','.join(vertices)))


    # Exercice
    # --------
    #
    # Construire le dictionnaire 'result' de la forme:
    #
    # result = {
    #   <id_visage_1>: {
    #       'anger': <resultat pour l'émotion "anger">,
    #	    'joy': <resultat pour l'émotion "joy">,
    #       'surprise': <resultat pour l'émotion "surprise">,
    #       'vertices': [
    #            (<vertex1>), # Tuple
    #            (<vertex2>),
    #            (<vertex3>),
    #            (<vertex4>)
    #            ]
    #       },
    #   <id_visage_2>: {
    #       'anger': <resultat pour l'émotion "anger">,
    #	    'joy': <resultat pour l'émotion "joy">,
    #       'surprise': <resultat pour l'émotion "surprise">,
    #       'vertices': [
    #            (<vertex1>),
    #            (<vertex2>),
    #            (<vertex3>),
    #            (<vertex4>)
    #            ]
    #       },
    #
    #   ...
    
    save_results(filename, result, extension='.faces.json')
# [END functions_detect_faces]
