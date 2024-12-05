from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import os
import warnings
import re 
from random import seed, randint

seed(7777) # pentru noroc (hai cu septarul!!!!!!!!!!!!!!!!)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_info(filename):
    match = re.match(r"([a-zA-Z]+)(\d+)\.\w+", filename)
    if match:
        return (match.group(1), int(match.group(2)))
    else:
        return None

def compute_distance(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2).item()

def list_files_in_folder(folder_path):
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(file_path)
    return files

def compute_image_tensors(image_files, batch_size = 32, resize = None):
    global device
    print(device)
    processor = AutoProcessor.from_pretrained('facebook/dinov2-large')
    model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
    model.eval()
    image_tensors = {}
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i : i + batch_size]
        images = []
        
        for img_path in batch_files:
            print("proc", img_path)
            img = Image.open(img_path).convert("RGB")
            if resize is not None:
                img = img.resize(resize, Image.Resampling.LANCZOS) 
            images.append(img)

        inputs = processor(images=images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        for idx, image_path in enumerate(batch_files):
            image_tensors[os.path.basename(image_path)] = outputs['pooler_output'][idx]

    return image_tensors

if __name__ == "__main__":

    image_paths = list_files_in_folder("data")
    image_tensors = compute_image_tensors(image_paths, batch_size = 32, resize = (500, 500))

    data = {}
    for f in image_tensors: 
        (name, id) = extract_info(f)
        if name not in data:
            data[name] = []
        data[name].append(image_tensors[f])


    total_data = 0
    for i in data:
        total_data += len(data[i])

    def test(dim_base):

        base_data = {}
        test_data = {}

        for i in data:
            base_now = min(len(data[i]), dim_base) 
            base_data[i] = data[i][: base_now]
            test_data[i] = data[i][base_now:]
        
        def compute_avg_distance(type, tensor):
            avg = 0
            for t in base_data[type]:
                avg += 1 / len(base_data[type]) * compute_distance(t, tensor)
            return avg 

        matrix = {}
        zic = {}

        for rl in data:
            zic[rl] = 0
            for pred in data:
                matrix[(rl, pred)] = 0


        for i in data:
            for guy in data[i]: 
                best = ""
                for type in data:
                    if best == "" or compute_avg_distance(type, guy) < compute_avg_distance(best, guy):
                        best = type
                assert (i, best) in matrix
                matrix[(i, best)] += 1
                zic[best] += 1
        
        for a in data:
            # din bayes avem P(este A | zic ca este A) = (P(zic ca este A | este A) * P(este A)) / P(zic ca este A)
            #                  p                       =               p1           *     p2     /        p3
            p1 = matrix[(a, a)] / len(data[a])
            p2 = len(data[a]) / total_data
            p3 = zic[a] / total_data
            if p3 == 0:
                print("nu zic niciodata ca este", a)
            else:
                p = p1 * p2 / p3
                print(dim_base, a, '------------->', p, p1)

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    for type in data:  
        for i in range(len(data[type])):
            data[type][i] = torch.cat([torch.tensor([0]).to(device), torch.cumsum(data[type][i], dim = 0)])
        
    def has_feature(featureInfo, tensorPartialSum):
        (st, dr, tresh) = featureInfo
        return tensorPartialSum[dr] - tensorPartialSum[st - 1] <= tresh

    def compute_avg_partialSum(st, dr):
        avg = 0
        total = 0
        for type in data:
            cntnow = 0
            for tensor in data[type]:
                total += 1
                avg += (tensor[dr] - tensor[st - 1])
                cntnow += 1
                #if cntnow >= 5:
                #    break
        return avg / total
    
    features = []
    
    def generate_random_pair(st, dr):
        l = randint(st, dr)
        
        r = randint(l, dr)

        return (l, l)


    for i in range(15):
        l = randint(1, 1024)
        dim = randint(1, 10)
        features.append(generate_random_pair(l, min(1024, l + dim)))
    features.append((1, 1024))

    features2 = []

    for i in range(len(features)):
        (st, dr) = features[i]
        avg = compute_avg_partialSum(st, dr)
        features2.append((st, dr, avg))

    features = features2
    

    p_sunt_categ = {}
    p_am_feature_f = {}

    pmatrix = {}
    p_categ_stiind_feature = {}

    for f in features:
        p_am_feature_f[f] = 0
        for categ in data:
            pmatrix[(f, categ)] = 0
            p_categ_stiind_feature[(categ, f)] = 0

    for categ in data:
        p_sunt_categ[categ] = len(data[categ]) / total_data
        for tensor in data[categ]:
            for f in features:
                has = has_feature(f, tensor)
                if has:
                    p_am_feature_f[f] += 1 / total_data 
                    pmatrix[(f, categ)] += 1 / len(data[categ])
                #print(has_feature(f, tensor))


    for f in features:
        sumrow = 0
        for categ in data:
            # vreau P(sunt din categ | am feature-ul f) = (P(am feature-ul f | sunt din categ) * P(sunt din categ))    / P(am feature-ul f)
            # p                                         = pmatrix[(f, categ)]                   * p_sunt_categ[categ]  / p_am_feature_f[f]
            p = pmatrix[(f, categ)] * p_sunt_categ[categ] / p_am_feature_f[f]
            #print(categ, f, p)
            sumrow += p
            p_categ_stiind_feature[(categ, f)] = p
            pass
        print("sumrow =", sumrow, "(verific sa fie foarte aproape de 1)") # trebuie sa fie foarte aproape de 1

    def predict(tensor):
        has = []
        for i in range(len(features)):
            has.append(has_feature(features[i], tensor))
        
        cn = ""
        prob = -1

        for categ in data:
            pcateg = 1
            for i in range(len(features)):
                if has[i]:
                    pcateg *= p_categ_stiind_feature[(categ, features[i])]
                else:
                    pcateg *= (1 - p_categ_stiind_feature[(categ, features[i])])
            if pcateg > prob:
                cn = categ 
                prob = pcateg
        return cn 
    
    good = 0
    total = 0
    for categ in data:
        for tensor in data[categ]:
            print("predict:", predict(tensor), "| real:", categ)
            good += (predict(tensor) == categ)
            total += 1
    print("pgood =", good / total)

    for i in data:
        
        print(i, len(data[i]))



'''
bayes:
P(A | B) = (P(B | A) * P(A)) / P(B)


'''