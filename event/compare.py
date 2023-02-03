from model import AudioCLIP
def check(model1, model2):
    model2 = model2.state_dict()
    model1 = model1.state_dict()
    for name in model1:
        p1 = model1[name]
        if name.split('.')[0] == 'audio':
            continue
        elif name in model2:
            print(name)
            p2 = model2[name]
            print((p1 == p2).all())

MODEL_FILENAME = 'AudioCLIP-Partial-Training.pt'
model1 = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}')
model2 = AudioCLIP()
print(check(model1, model2))