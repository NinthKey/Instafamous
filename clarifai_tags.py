from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import pickle

tag_count = {}
response = None

def get_tags(model, image_url, threshold = 0.9):
    global tag_count
    global response
    response = model.predict_by_url(image_url)
    tags = []
    for concept in response['outputs'][0]['data']['concepts']:
        if concept['value'] >= threshold:
            tags.append(concept['name'])
            if tag_count[concept['name']] == None:
                tag_count[concept['name']] = 1
            else:
                tag_count[concept['name']] += 1
    return tags

def get_all_image_tags(model, images, tag_count = {}, threshold = 0.90):
    img_tags = {}
    global response
    response = model.predict(images)
    for output in response['outputs']:
        tags = []
        url = output['input']['data']['image']['url']
        for concept in output['data']['concepts']:
            if concept['value'] >= threshold:
                tags.append(concept['name'])
                if concept['name'] in tag_count:
                    tag_count[concept['name']] += 1
                else:
                    tag_count[concept['name']] = 1
                    
        img_tags[url] = tags
    return img_tags, tag_count

def get_images_tags(model, images, threshold = 0.90):
    tag_list = []
    global response
    
    index = 0
    end = len(images);
    
    while index < end and index + 128 < 5000:
        if end - index > 128:
            response = model.predict(images[index:index+128])
            index += 128
        else:
            response = model.predict(images[index:end])
            index = end
        for output in response['outputs']:
            tags = []
            for concept in output['data']['concepts']:
                if concept['value'] >= threshold:
                    tags.append(concept['name'])               
            tag_list.append(tags)
            
    return tag_list


# username, followers, following, nposts, ntags, time, ...n features..., likes

def preprocess_data(model, users, input_tags):
    result = []
    images = []
    nrow = 0
    for user in users:
        for post in user['post']:
            images.append(ClImage(url=post['img']))
    tag_list = get_images_tags(model, images, 0.8)
    for user in users:
        for post in user['post']:
            row = [user['user'], user['followers'], user['following'], user['num_of_posts'], post['numberOfTag'], post['timeOfDay']];
            tags = tag_list[nrow]
            nrow += 1
            for tag in input_tags:
                if tag in tags:
                    row.append(1)
                else:
                    row.append(0)
            row.append(post['likes'])
            result.append(row)
            if nrow == len(tag_list)-1:
                break
        if nrow == len(tag_list)-1:
            break
    return result
    

def load_obj(name):
	with open(name + '.pkl', 'rb+') as f:
		return pickle.load(f)

def save_obj(obj, name):
	with open(name + '.pkl', 'wb+') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


app = ClarifaiApp(api_key='e03f648f9f84485e96008090c27eacd3')
model = app.models.get('general-v1.3')
"""
images = []
#for i in range(1, 14):
#    url = "https://samples.clarifai.com/demo-%03d.jpg" % i
#    img = ClImage(url=url)
#    images.append(img)
urls = load_obj('temp')
for url in urls:
    img = ClImage(url=url)
    images.append(img)
    
img_tags, tag_count = get_all_image_tags(model, images)

save_obj(tag_count, 'tag_count')
save_obj(img_tags, 'img_tags')


rmkeys = []
for key, value in tag_count.items():
    if value < 4:
        rmkeys.append(key)
for key in rmkeys:
    del tag_count[key]

save_obj(tag_count, '4orabovetags')
""" 
tag_count = load_obj('4orabovetags')
users = load_obj('temp')
users = users[143:]


X = preprocess_data(model, users, tag_count)

save_obj(X, 'input2')

