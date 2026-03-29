# 1. ベースイメージ: Ubuntu 22.04 + ROS 2 Humble Desktop
FROM osrf/ros:humble-desktop

# 2. apt-getの対話プロンプトを無効化
ENV DEBIAN_FRONTEND=noninteractive

# 3. 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ignition-fortress \
    ros-humble-ros-ign \
    && rm -rf /var/lib/apt/lists/*

# 4. コンテナ内での作業ディレクトリ（フォルダ）を /app に設定
WORKDIR /app

# 5. プロジェクトファイルのコピー
COPY . /app

# 6. 環境変数の自動読み込み
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
