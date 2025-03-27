# src/pipelineA/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20):
    batch_size, num_dims, num_points = x.size()
    idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature



class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x



class DGCNN_seg(nn.Module):
    def __init__(self, k=20, emb_dims=1024, num_classes=2, dropout=0.2, categorical_dim=1):
        super(DGCNN_seg, self).__init__()
        self.k = k
        self.transform_net = Transform_Net()

        # self.categorical_dim = categorical_dim
        # 编码器部分 - 与分类模型相同的EdgeConv层
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                           nn.BatchNorm2d(64),
                           nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(negative_slope=0.2))

        # 特征聚合
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 解码器部分 - 将全局特征扩散到每个点
        self.conv6 = nn.Sequential(
            nn.Conv1d(2112, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.dp1 = nn.Dropout(p=dropout)
        
        # 分割头 - 预测每个点的类别
        self.conv8 = nn.Conv1d(256, num_classes, kernel_size=1, bias=True)

        # 分类向量（categorical vector）的处理MLP
        self.cat_mlp = nn.Sequential(
            nn.Linear(categorical_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, categorical_vector):
        batch_size = x.size(0)
        num_points = x.size(1)
        x = x.permute(0, 2, 1)  # (B, 3, N)

        x0 = get_graph_feature(x, k=self.k)
        t = self.transform_net(x0)
        x = x.permute(0, 2, 1)  # (B, 3, N)
        x = torch.bmm(x, t)
        x = x.permute(0, 2, 1)



        # 编码器 - 提取边缘特征
        x1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1)[0]

        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1)[0]

        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1)[0]

        # 特征聚合
        x_cat = torch.cat((x1, x2, x3), dim=1)  # (B, 512, N)
        x4 = self.conv5(x_cat)  # (B, emb_dims, N)
        
        # 全局特征
        x_global = F.adaptive_max_pool1d(x4, 1).expand(-1, -1, num_points)  # (B, emb_dims, N)
        
        # 分类向量处理并扩展
        cat_vec = self.cat_mlp(categorical_vector).unsqueeze(2).expand(-1, -1, num_points)  # (B, 64, N)
    
        # 拼接全局特征和分类向量
        x = torch.cat((x_global, cat_vec), dim=1)  # (B, emb_dims + 64, N)

        # x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x4), dim=1)  # (batch_size, 1088 + emb_dims, num_points)

        # 解码器 - 将全局特征扩散到每个点

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.dp1(x)
        x_seg = self.conv8(x)  # (batch_size, num_classes, num_points)



        return x_seg
    
def test_forward():
    model = DGCNN_seg(k=20, emb_dims=1024, num_classes=2, dropout=0.5, categorical_dim=2)
    # 随机生成一个 batch 的点云数据，点数量设为1024，点的维度为3
    x = torch.randn(2, 1024, 3)
    # 随机生成对应的分类向量
    cat = torch.randn(2, 2)
    output = model(x, cat)
    print("输出形状:", output.shape)

if __name__ == "__main__":
    test_forward()