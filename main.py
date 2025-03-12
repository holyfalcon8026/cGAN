import streamlit as st
import torch
import torch.nn as nn

# モデル定義
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4  # 28 // 4 = 7
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのロード
latent_dim = 10
n_classes = 10
generator = Generator(latent_dim, n_classes).to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# Streamlit UI
st.title("Conditional GAN 画像生成")
st.write("ノイズとラベルを入力して手書き数字を生成")

# ユーザー入力
z_input = [st.slider(f"ノイズ {i+1}", -1.0, 1.0, 0.0) for i in range(latent_dim)]
label_input = st.selectbox("生成する数字 (0-9)", list(range(10)))

if st.button("画像生成"):
    noise = torch.tensor([z_input], dtype=torch.float32).to(device)
    label = torch.tensor([label_input], dtype=torch.long).to(device)

    with torch.no_grad():
        generated_img = generator(noise, label)

    generated_img = generated_img.squeeze().cpu().numpy()

    st.image(generated_img, caption="生成された画像", width=200, clamp=True)
