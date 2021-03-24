import matplotlib.pyplot as plt
import tensorly as tl
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dttd_utils import *


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, middle_size):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, middle_size),
        )
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(middle_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def Train(autoencoder, train_loader, alpha, gamma, epoch=EPOCH_TRAIN):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=lambda_reg)
    loss_func = nn.MSELoss()

    for epoch in range(epoch):
        for step, D in enumerate(train_loader):
            P, U = D
            encoded, decoded = autoencoder(Noise(P, 0.3))

            loss1 = alpha * loss_func(decoded, P)
            I = torch.eye(encoded.size(0), device='cuda')
            loss2 = alpha * loss_func(torch.mm(encoded, torch.t(encoded)), I)
            loss3 = gamma * loss_func(encoded, U)

            loss = loss1 + loss2 + loss3
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients


if __name__ == '__main__':

    for s_t in source_target:
        source = s_t[0]
        target = s_t[1]
        for startIteration in startIteration_list:
            for randomNum in randomNum_list:
                for align_status in [True]:
                    thelist = []
                    stat_list_rmse = []
                    stat_list_mae = []
                    device = torch.device(device_str)
                    loss_rmse_list = np.zeros(epochs_whole)
                    loss_mae_list = np.zeros(epochs_whole)
                    loss_time = np.arange(epochs_whole)
                    minRmseL = []
                    minMaeL = []
                    maxHrL = []
                    maxNdcgL = []
                    r = 7
                    HrK = np.zeros((r, 5))
                    NdcgK = np.zeros((r, 5))
                    minList = []
                    perList = [0.4, 0.20, 0.05]
                    perList = [0.20]
                    for percent in perList:
                        classes = (iteration + startIteration) * 5
                        features = classes
                        filename = classes
                        minRmse = 5
                        minMae = 5
                        maxHr = 0
                        maxNdcg = 0
                        parameter = 'latentFactor%s_%s%s%s' % (classes, randomNum, align_status, percent)
                        trainAddress = '%s/%s' % (rootdir, source)
                        (R_s_u, C_s_u, R_s_u2, C_s_u2, P_s_u, P_s_v, input_size_p_s_u, input_size_p_s_v, input_size_x_s,
                         input_size_x_s_u,
                         input_size_x_s_v, s_u_num, s_v_num, hidden_size_s_u, hidden_size_s_v, testData_s,
                         Criteria_s_Num) = ReadData(trainAddress,
                                                    percent, randomNum, device)
                        trainAddress = '%s/%s' % (rootdir, target)
                        (R_t_u, C_t_u, R_t_u2, C_t_u2, P_t_u, P_t_v, input_size_p_t_u, input_size_p_t_v, input_size_x_t,
                         input_size_x_t_u,
                         input_size_x_t_v, t_u_num, t_v_num, hidden_size_t_u, hidden_size_t_v, testData_t,
                         Criteria_t_Num) = ReadData(trainAddress,
                                                    percent, randomNum, device)
                        data_dir = '%s/DTTDtarget_%s_%s/%s' % (
                            trainAddress, target, source, parameter) + '/'
                        if not os.path.exists(data_dir):
                            os.makedirs(data_dir)
                        path_t = data_dir + 'autoencoder_s_u' + 'latent' + str(features) + '.pkl'
                        print('TrainPre')
                        if os.path.exists(path_t):
                            autoencoder_s_u = torch.load(path_t, map_location=device_str)
                        else:
                            autoencoder_s_u = AutoEncoder(input_size_p_s_u, hidden_size_s_u, features).to(device)
                        path_t = data_dir + 'autoencoder_s_v' + 'latent' + str(features) + '.pkl'
                        if os.path.exists(path_t):
                            autoencoder_s_v = torch.load(path_t, map_location=device_str)
                        else:
                            autoencoder_s_v = AutoEncoder(input_size_p_s_v, hidden_size_s_v, classes).to(device)
                        path_t = data_dir + 'autoencoder_t_u' + 'latent' + str(features) + '.pkl'
                        if os.path.exists(path_t):
                            autoencoder_t_u = torch.load(path_t, map_location=device_str)
                        else:
                            autoencoder_t_u = AutoEncoder(input_size_p_t_u, hidden_size_t_u, features).to(device)
                        path_t = data_dir + 'autoencoder_t_v' + 'latent' + str(features) + '.pkl'
                        if os.path.exists(path_t):
                            autoencoder_t_v = torch.load(path_t, map_location=device_str)
                        else:
                            autoencoder_t_v = AutoEncoder(input_size_p_t_v, hidden_size_t_v, classes).to(device)

                        print('TrainStacked')
                        (U_s, V_s, Criteria_s, Core_s) = ReadInit(Criteria_s_Num, input_size_x_s, features, data_dir,
                                                                  percent, 's',
                                                                  device, c_r)
                        U_s_state = {}
                        V_s_state = {}
                        Criteria_s_state = {}
                        H_s_state_L = []
                        for i in range(Criteria_s.shape[0]):
                            H_s_state_L.append({})

                        (U_t, V_t, Criteria_t, Core_t) = ReadInit(Criteria_t_Num, input_size_x_t, features, data_dir,
                                                                  percent, 't',
                                                                  device, c_r)
                        U_t_state = {}
                        V_t_state = {}
                        Criteria_t_state = {}

                        print('pretrain')
                        factors = [U_s, V_s, Criteria_s]
                        tl.set_backend('pytorch')
                        R_s_u_loader = DataLoader(TensorDataset(P_s_u, U_s), batch_size=batch_size_s, shuffle=True,
                                                  num_workers=0)
                        R_s_v_loader = DataLoader(TensorDataset(P_s_v, V_s), batch_size=batch_size_s, shuffle=True,
                                                  num_workers=0)
                        Train(autoencoder_s_u, R_s_u_loader, alpha_s, rho_s, epoch=EPOCH_PRETRAIN)
                        Train(autoencoder_s_v, R_s_v_loader, beta_s, gamma_s, epoch=EPOCH_PRETRAIN)

                        R_t_u_loader = DataLoader(TensorDataset(P_t_u, U_t), batch_size=batch_size_t, shuffle=True,
                                                  num_workers=0)
                        R_t_v_loader = DataLoader(TensorDataset(P_t_v, V_t), batch_size=batch_size_t, shuffle=True,
                                                  num_workers=0)
                        Train(autoencoder_t_u, R_t_u_loader, alpha_t, rho_t, epoch=EPOCH_PRETRAIN)
                        Train(autoencoder_t_v, R_t_v_loader, beta_t, gamma_t, epoch=EPOCH_PRETRAIN)

                        print('train')
                        best_model = []

                        for i in range(epochs_whole):
                            R_s_u_loader = DataLoader(TensorDataset(P_s_u, U_s), batch_size=batch_size_s, shuffle=True,
                                                      num_workers=0)
                            R_s_v_loader = DataLoader(TensorDataset(P_s_v, V_s), batch_size=batch_size_s, shuffle=True,
                                                      num_workers=0)
                            Train(autoencoder_s_u, R_s_u_loader, alpha_s, rho_s, epoch=EPOCH_TRAIN)
                            Train(autoencoder_s_v, R_s_v_loader, beta_s, gamma_s, epoch=EPOCH_TRAIN)

                            R_t_u_loader = DataLoader(TensorDataset(P_t_u, U_t), batch_size=batch_size_t, shuffle=True,
                                                      num_workers=0)
                            R_t_v_loader = DataLoader(TensorDataset(P_t_v, V_t), batch_size=batch_size_t, shuffle=True,
                                                      num_workers=0)
                            Train(autoencoder_t_u, R_t_u_loader, alpha_t, rho_t, epoch=EPOCH_TRAIN)
                            Train(autoencoder_t_v, R_t_v_loader, beta_t, gamma_t, epoch=EPOCH_TRAIN)

                            with torch.no_grad():
                                M_s_u = autoencoder_s_u(P_s_u)[0]  # def forward(self,x,u_or_v):
                                M_s_v = autoencoder_s_v(P_s_v)[0]
                                M_t_u = autoencoder_t_u(P_t_u)[0]  # def forward(self,x,u_or_v):
                                M_t_v = autoencoder_t_v(P_t_v)[0]

                                if align_status:
                                    for j in range(epochs_uv):
                                        (U_s, V_s, Criteria_s, Core_s) = UVCupdate(U_s, V_s, Criteria_s, U_s_state,
                                                                                   V_s_state,
                                                                                   Criteria_s_state,
                                                                                   u_rate, v_rate, c_rate, R_s_u, C_s_u,
                                                                                   device, M_s_u,
                                                                                   M_s_v,
                                                                                   Core_s)
                                        Core_t = Core_s
                                        (U_t, V_t, Criteria_t, Core_t) = UVCupdate_t(U_t, V_t, Criteria_t, U_t_state,
                                                                                     V_t_state,
                                                                                     Criteria_t_state,
                                                                                     u_rate, v_rate, c_rate, R_t_u,
                                                                                     C_t_u, device, M_t_u,
                                                                                     M_t_v,
                                                                                     Core_t)
                                else:
                                    (U_s, V_s, Criteria_s, Core_s) = UVCupdate(U_s, V_s, Criteria_s, U_s_state,
                                                                               V_s_state,
                                                                               Criteria_s_state,
                                                                               u_rate, v_rate, c_rate, R_s_u, C_s_u,
                                                                               device, M_s_u,
                                                                               M_s_v,
                                                                               Core_s)
                                    (U_t, V_t, Criteria_t) = UVCupdate_t(U_t, V_t, Criteria_t, U_t_state, V_t_state,
                                                                         Criteria_t_state,
                                                                         u_rate, v_rate, c_rate, R_t_u, C_t_u, device,
                                                                         M_t_u,
                                                                         M_t_v,
                                                                         Core_t)
                                    Core_t = Core_s

                            if align_status:
                                U_t, V_t, Criteria_t = alignment(U_s, V_s, Criteria_s, U_t, V_t, Criteria_t)

                            print('The epoch ' + str(i) + ' is finished.')

                            net_path = data_dir + 'autoencoder_s_u' + 'latent' + str(features) + '.pkl'
                            torch.save(autoencoder_s_u, net_path)
                            net_path = data_dir + 'autoencoder_s_v' + 'latent' + str(features) + '.pkl'
                            torch.save(autoencoder_s_v, net_path)

                            path_t = data_dir + str(percent) + 'latent' + str(features) + 'U_s' + '.pkl'
                            torch.save(U_s, path_t)
                            path_t = data_dir + str(percent) + 'latent' + str(features) + 'V_s' + '.pkl'
                            torch.save(V_s, path_t)
                            path_t = data_dir + str(percent) + 'latent' + str(features) + 'Criteria_s' + '.pkl'
                            torch.save(Criteria_s, path_t)

                            f = open(data_dir + 'loss.log', 'a')
                            print('Epochs:' + str(i), file=f)
                            print('Epochs:' + str(i))

                            ratingErrorTrain = getRatingError(Core_s, U_s, V_s, Criteria_s, R_s_u, C_s_u)

                            loss_rmse_s = torch.norm(ratingErrorTrain) / torch.sqrt(torch.sum(C_s_u))
                            loss_mae_s = torch.sum(torch.abs(ratingErrorTrain)) / torch.sum(C_s_u)
                            print('Loss at Train Data is rmse:' + str(loss_rmse_s.cpu().data.numpy()) + ',mae:' + str(
                                loss_mae_s.cpu().data.numpy()), file=f)
                            print('Loss at Train Data is rmse:' + str(loss_rmse_s.cpu().data.numpy()) + ',mae:' + str(
                                loss_mae_s.cpu().data.numpy()))

                            ratingErrorTrain = getRatingError(Core_s, U_s, V_s, Criteria_s, R_s_u2, C_s_u2)
                            loss_rmse_s = torch.norm(ratingErrorTrain) / torch.sqrt(torch.sum(C_s_u2))
                            loss_mae_s = torch.sum(torch.abs(ratingErrorTrain)) / torch.sum(C_s_u2)
                            print('Loss at Source Test Data is rmse:' + str(
                                loss_rmse_s.cpu().data.numpy()) + ',mae:' + str(
                                loss_mae_s.cpu().data.numpy()), file=f)
                            print('Loss at Source Test Data is rmse:' + str(
                                loss_rmse_s.cpu().data.numpy()) + ',mae:' + str(
                                loss_mae_s.cpu().data.numpy()))

                            print('', file=f)

                            ratingErrorTest, stat_list_temp = getRatingError_stat(Core_t, U_t, V_t, Criteria_t, R_t_u2,
                                                                                  C_t_u2)

                            loss_rmse = torch.norm(ratingErrorTest) / torch.sqrt(torch.sum(C_t_u2))
                            loss_mae = torch.sum(torch.abs(ratingErrorTest)) / torch.sum(C_t_u2)
                            (preRating) = getPreRating(U_t, V_t, Criteria_t, Core_t).to(
                                torch.device("cpu")).detach().numpy()
                            thre = 10
                            _K = 10
                            (hr, ndcg) = getHR_NDCGS(trainAddress, testData_t, thre, _K, preRating)

                            print(
                                'Loss at Target Test Data is rmse:' + str(loss_rmse.cpu().data.numpy()) + ',mae:' + str(
                                    loss_mae.cpu().data.numpy()) + 'HR = %.4f, NDCG = %.4f,' % (hr, ndcg), file=f)
                            print(
                                'Loss at Target Test Data is rmse:' + str(loss_rmse.cpu().data.numpy()) + ',mae:' + str(
                                    loss_mae.cpu().data.numpy()) + 'HR = %.4f, NDCG = %.4f,' % (hr, ndcg))

                            loss_rmse_list[i] = loss_rmse
                            plt.plot(loss_time, loss_rmse_list, 'r')
                            plt.ylim(0.5, 1.1)
                            plt.savefig(data_dir + 'loss_rmse.jpg')
                            plt.close(0)
                            loss_mae_list[i] = loss_mae
                            plt.plot(loss_time, loss_mae_list, 'r')
                            plt.ylim(0.5, 1.1)
                            plt.savefig(data_dir + 'loss_mae.jpg')
                            plt.close(1)
                            f.close()

                            if float(loss_rmse) < minRmse:
                                minRmse = float(loss_rmse)
                                stat_list_rmse = stat_list_temp
                            if float(loss_mae) < minMae:
                                minMae = float(loss_mae)
                                stat_list_mae = stat_list_temp
                            if maxHr < hr:
                                maxHr = float(hr)
                            if maxNdcg < ndcg:
                                maxNdcg = float(ndcg)

                            for i in range(5):
                                thre = (i + 1) * 2
                                _K = (i + 1) * 2
                                (hr, ndcg) = getHR_NDCGS(trainAddress, testData_t, thre, _K, preRating)
                                HrK[iteration][i], NdcgK[iteration][i] = getBigger(HrK[iteration][i],
                                                                                   NdcgK[iteration][i], hr, ndcg)

                        print('end')
                        print(minRmse, minMae)
                        minRmseL.append(minRmse)
                        minMaeL.append(minMae)
                        maxHrL.append(maxHr)
                        maxNdcgL.append(maxNdcg)
                        thelist.append(minRmseL)
                        thelist.append(minMaeL)
                        thelist.append(maxHrL)
                        thelist.append(maxNdcgL)
                        listlogAd = '%s/listLog_DTTD.txt' % trainAddress

                        file_handle = open(listlogAd, mode='w')

                        for l in thelist:
                            temp = ''
                            for j in l:
                                temp = temp + '%.4f' % j + '\t,'
                            file_handle.write(temp)
                            file_handle.write('\n')
                            print(temp)

                        file_handle.write('HrK is \n')
                        for l in HrK:
                            temp = ''
                            for j in l:
                                temp = temp + '%.4f' % j + '\t,'
                            file_handle.write(temp)
                            file_handle.write('\n')
                            print(temp)

                        file_handle.write('NdcgK is \n')
                        for l in NdcgK:
                            temp = ''
                            for j in l:
                                temp = temp + '%.4f' % j + '\t,'
                            file_handle.write(temp)
                            file_handle.write('\n')
                            print(temp)
                        file_handle.close()

                        with open(data_dir + 'stat.txt', 'w') as f:
                            f.write('Core_tensor: ' + str(5 * startIteration) + '\n')
                            f.write('\t randomnum: ' + str(randomNum) + '\n')
                            f.write('\t minRMSE: ' + str(minRmse) + ', minMAE: ' + str(minMae) + '\n')
                            f.write('\t\t' +
                                    'rmse: %d, %d, %d, %d, %d \n' % (stat_list_rmse[1] - stat_list_rmse[0],
                                                                     stat_list_rmse[2] - stat_list_rmse[1],
                                                                     stat_list_rmse[3] - stat_list_rmse[2],
                                                                     stat_list_rmse[4] - stat_list_rmse[3],
                                                                     stat_list_rmse[5] - stat_list_rmse[4]))
                            f.write('\t\t' +
                                    'mae: %d, %d, %d, %d, %d \n' % (stat_list_mae[1] - stat_list_mae[0],
                                                                    stat_list_mae[2] - stat_list_mae[1],
                                                                    stat_list_mae[3] - stat_list_mae[2],
                                                                    stat_list_mae[4] - stat_list_mae[3],
                                                                    stat_list_mae[5] - stat_list_mae[4]))
                            f.write('\n')
                        torch.cuda.empty_cache()
                        torch.cuda.init()
