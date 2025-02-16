import os
import discord
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from scipy.spatial.distance import cosine
import asyncio
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv

load_dotenv()

# 봇 설정
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix='!', intents=intents)

# 미리 저장된 이미지들이 들어 있는 폴더 경로
STORED_IMAGES_FOLDER = './images'  # 여기에 저장된 이미지들이 있는 폴더 경로를 입력

server_languages = {}

@client.tree.command()
async def set_server(interaction: discord.Interaction, language: str):
    """서버 언어 설정 명령어"""
    server_id = interaction.guild.id

    if language.lower() not in ['korean', 'english']:
        await interaction.response.send_message("Invalid language. Please use 'korean' or 'english'.")
        return

    # 언어에 따른 역할 이름 설정
    if language.lower() == 'korean':
        role_name = "구독자"
    elif language.lower() == 'english':
        role_name = "Subscriber"

    # 서버에 언어 정보 저장
    server_languages[server_id] = role_name

    await interaction.response.send_message(f"Server language set to {language}. The role name is set to '{role_name}'.")

# 미리 저장된 이미지 불러오기
def load_image(image_path):
    return Image.open(image_path)

# 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # 배치 차원을 추가

# 모델 로드 (사전 학습된 ResNet 모델 사용)
model = models.resnet18(pretrained=True)
model.eval()  # 평가 모드로 설정

# 이미지의 특징 벡터를 추출하는 함수
def extract_features(image):
    image = preprocess_image(image)
    with torch.no_grad():
        features = model(image)  # 모델을 통해 특징 추출
        features = features.squeeze()  # 배치 차원을 제거하여 1D 벡터로 변환
    return features

# 폴더에 있는 모든 이미지의 특징 벡터를 추출
def load_stored_images_features(folder_path):
    features_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = load_image(image_path)
            features = extract_features(image)
            features_list.append((filename, features))
    return features_list

# 이미지 유사도 계산
def calculate_similarity(features1, features2):
    # Cosine similarity 계산 (1D 벡터로 변환 후)
    return 1 - cosine(features1.numpy(), features2.numpy())

# Discord 이벤트 핸들러
@client.event
async def on_ready():
    await client.tree.sync()
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # 특정 채널에서만 작동
    if message.channel.id == 1326814475906449440 or message.channel.id == 1285362467173695578:
        if message.attachments:
            for attachment in message.attachments:
                if attachment.filename.endswith(('png', 'jpg', 'jpeg')):

                    # 첨부된 이미지를 다운로드
                    img_data = await attachment.read()
                    img = Image.open(io.BytesIO(img_data))

                    # 미리 저장된 이미지들의 특징 벡터를 불러오기
                    stored_features_list = load_stored_images_features(STORED_IMAGES_FOLDER)

                    # 업로드된 이미지의 특징 벡터 추출
                    uploaded_features = extract_features(img)

                    # 저장된 이미지들과 비교하여 가장 유사한 이미지 찾기
                    max_similarity = -1  # 초기값은 가장 낮은 유사도
                    most_similar_image = None
                    for filename, stored_features in stored_features_list:
                        similarity = calculate_similarity(uploaded_features, stored_features)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_image = filename

                    # 유사도가 임계값 이상이면 콘솔에 메시지 출력
                    await message.reply("Processing image...")
                    await asyncio.sleep(2)

                    if max_similarity == 1.0:
                        await message.reply("Image is exactly the same as the stored image. You are likely to have copied another user's image. We can't verify you. \nDo you think it's a mistake? Contact the server owner!")
                    else:
                        await message.reply(f"Uploaded image is {max_similarity*100:.2f}% similar to {most_similar_image}")

                    if max_similarity >= 0.7:
                        await asyncio.sleep(1)
                        # 서버의 역할 이름 가져오기
                        role_name = server_languages.get(message.guild.id, None)
                        if role_name:
                            role = discord.utils.get(message.guild.roles, name=role_name)
                            if role:
                                # 역할 부여
                                await message.author.add_roles(role)
                                await message.reply(f"# Verified!\nYou got the `{role_name}` role.")

                                # 언어 설정에 따라 삭제할 역할 결정
                                if role_name == "구독자":  # Korean 설정
                                    remove_role_name = "인증-안함"
                                else:  # English 설정
                                    remove_role_name = "Unverified"

                                # 역할 제거
                                remove_role = discord.utils.get(message.guild.roles, name=remove_role_name)
                                if remove_role and remove_role in message.author.roles:
                                    await message.author.remove_roles(remove_role)
                                    await message.reply(f"The `{remove_role_name}` role removed from user.")
                            else:
                                await message.reply(f"Role '{role_name}' not found.")
                        else:
                            await message.reply("No language setting found for this server.")
                    else:
                        await message.reply(f"Not verified. Your image's similarity is not high enough\n Your score: {max_similarity*100:.2f}%\n Required score: 70%\nPlease wait for an admin to check your image. It usually takes less than 24 hours.")

token = os.getenv("DISCORD_TOKEN")
client.run(token)
