import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

device = [0,1]

from model import CreateModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm


torch.manual_seed(0)


# Data
tick = torch.arange(0., 256.)/255.          #0~255
R, G, B = torch.meshgrid(tick,tick,tick)
# gamma 1.1
X = torch.stack((torch.reshape(R,(-1,1)), torch.reshape(G,(-1,1)), torch.reshape(B,(-1,1))),dim = 1)
X = torch.unsqueeze(X, 2)
Y = torch.pow(X, 1/1.1)

X -= 0.5
Y -= 0.5

dataset = Data.TensorDataset(X, Y)
data_iter = Data.DataLoader(dataset,batch_size=256*256, shuffle=True, num_workers=32)






MODEL,__ = CreateModel([4, 8, 16, 24, 40])
# MODEL = nn.DataParallel(MODEL,device_ids=[0,1]).cuda()

MODEL = torch.nn.DataParallel(MODEL, device_ids=device)




MODEL = MODEL.to('cuda')

# if 1:
#     avgSAD = list()
#     maxSAD = list()
#     for idx, data in enumerate(data_iter):
#         X_, Y_ = data
#         Y_= Y_.to('cuda')

#         Y_pred = MODEL(X_)
#         Y_pred += 0.5
#         Y_pred = torch.clamp(Y_pred, 0., 1.)
#         Y_pred = torch.round(255*Y_pred)

#         Y_tmp = Y_
#         Y_tmp += 0.5
#         Y_tmp = torch.round(255*Y_tmp)

#         ABSDIFF = torch.abs(Y_pred-Y_tmp)
#         SAD = torch.sum(ABSDIFF, 1)

#         avgSAD.append(float(torch.mean(SAD)))
#         maxSAD.append(float(torch.max(SAD)))
#     print(f'avgSAD:{sum(avgSAD)/len(avgSAD):.4f}')
#     print(f'maxSAD:{max(maxSAD)}')

MODEL.load_state_dict(torch.load('./2_model_params_Epoch22585_0.000106394_1.pth'))

if 1:
    avgSAD = list()
    maxSAD = list()
    for idx, data in enumerate(data_iter):
        X_, Y_ = data
        Y_= Y_.to('cuda')

        Y_pred = MODEL(X_)
        Y_pred += 0.5
        Y_pred = torch.clamp(Y_pred, 0., 1.)
        Y_pred = torch.round(255*Y_pred)

        Y_tmp = Y_
        Y_tmp += 0.5
        Y_tmp = torch.round(255*Y_tmp)

        ABSDIFF = torch.abs(Y_pred-Y_tmp)
        SAD = torch.sum(ABSDIFF, 1)

        avgSAD.append(float(torch.mean(SAD)))
        maxSAD.append(float(torch.max(SAD)))

    avgSAD = sum(avgSAD)/len(avgSAD)
    maxSAD = max(maxSAD)
    print(f'avgSAD:{avgSAD:.4f}, maxSAD:{maxSAD}')




criterion = nn.MSELoss()
# criterion = nn.SmoothL1Loss

parameters = [p for p in MODEL.parameters() if p.requires_grad]
optimizer = optim.Adamax(parameters, lr=0.0001)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=50, verbose=True)

best_avgSAD = avgSAD
best_maxSAD = maxSAD


iters = len(data_iter)
for epoch in range(40000):
    with tqdm(total = len(dataset)) as epoch_pbar:
        epoch_pbar.set_description(f'Epoch {epoch}')

        acc_loss = 0.0
        for idx, data in enumerate(data_iter):
            X_, Y_ = data
            Y_= Y_.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = MODEL(X_)

            loss = criterion(outputs, Y_)
            loss.backward()
            optimizer.step()

            # Update progress bar description
            lr = optimizer.param_groups[0]['lr']
            acc_loss += loss
            avg_loss = acc_loss / (idx + 1)
            desc = f'Epoch {epoch+1}, loss {avg_loss:.2e}, lr {lr:.2e}'
            epoch_pbar.set_description(desc)
            epoch_pbar.update(X_.shape[0])
        scheduler.step()
        # scheduler.step(loss)


    if epoch % 5 == 4:
        avgSAD = list()
        maxSAD = list()
        for idx, data in enumerate(data_iter):
            X_, Y_ = data
            Y_= Y_.to('cuda')

            Y_pred = MODEL(X_)
            Y_pred += 0.5
            Y_pred = torch.clamp(Y_pred, 0., 1.)
            Y_pred = torch.round(255*Y_pred)

            Y_tmp = Y_
            Y_tmp += 0.5
            Y_tmp = torch.round(255*Y_tmp)

            ABSDIFF = torch.abs(Y_pred-Y_tmp)
            SAD = torch.sum(ABSDIFF, 1)

            avgSAD.append(float(torch.mean(SAD)))
            maxSAD.append(float(torch.max(SAD)))

        avgSAD = sum(avgSAD)/len(avgSAD)
        maxSAD = max(maxSAD)
        print(f'Epoch {epoch+1},avgSAD:{avgSAD:.4f}, maxSAD:{maxSAD}',end='\t')

        PTH = f'2_model_params_Epoch{epoch+1}_{avgSAD:.9f}_{maxSAD:.0f}.pth'
        if maxSAD < best_maxSAD:
            best_maxSAD = maxSAD
            best_avgSAD = avgSAD
            torch.save(MODEL.state_dict(), PTH)
            print(f'saving(best_maxSAD)... {PTH} ')
        elif maxSAD == best_maxSAD:
            if avgSAD < best_avgSAD:
                best_maxSAD = maxSAD
                best_avgSAD = avgSAD
                torch.save(MODEL.state_dict(), PTH)
                print(f'saving(best_avgSAD)... {PTH} ')
        else:
            print()





print('Finished Training')



MODEL.eval()

avgSAD = list()
maxSAD = list()
for idx, data in enumerate(data_iter):
    X_, Y_ = data
    Y_= Y_.to('cuda')

    Y_pred = MODEL(X_)
    Y_pred += 0.5
    Y_pred = torch.clamp(Y_pred, 0., 1.)
    Y_pred = torch.round(255*Y_pred)

    Y_tmp = Y_
    Y_tmp += 0.5
    Y_tmp = torch.round(255*Y_tmp)

    ABSDIFF = torch.abs(Y_pred-Y_tmp)
    SAD = torch.sum(ABSDIFF, 1)

    avgSAD.append(float(torch.mean(SAD)))
    maxSAD.append(float(torch.max(SAD)))
print(f'avgSAD:{sum(avgSAD)/len(avgSAD):.4f}')
print(f'maxSAD:{max(maxSAD)}')
