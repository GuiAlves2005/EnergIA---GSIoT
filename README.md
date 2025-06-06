## EnergIA---GSIoT

# Guilherme Alves de Lima RM:550433
# Pedro Guerra RM:99526


## Link Video -- https://youtu.be/RBhKvM3xOkM


🧠 Descrição do Problema
Durante quedas de energia, ambientes como hospitais e centros de controle podem se tornar perigosos, especialmente para pessoas com mobilidade reduzida ou deficiência visual. A ausência de iluminação compromete a visibilidade e impede a detecção de comportamentos de risco, como quedas, gestos de pedido de ajuda ou movimentos bruscos em locais perigosos. Além disso, equipes de segurança ou socorro têm dificuldade para agir rapidamente sem um sistema auxiliar automatizado.

Diante desse cenário, surge a necessidade de uma solução acessível, baseada em visão computacional, que funcione mesmo com iluminação reduzida (ou simulando esse contexto) e seja capaz de reconhecer gestos humanos ou padrões de movimento relevantes para acionar alertas e prevenir acidentes.

💡 Visão Geral da Solução
Desenvolvemos um sistema de detecção de gestos de emergência usando Python e MediaPipe, capaz de funcionar a partir de vídeos simulando ambientes escuros. A solução reconhece gestos como joinha ou punho fechado – previamente definidos como sinais de alerta ou está tudo ok – e gera uma resposta automática na interface, que poderia futuramente acionar alarmes ou enviar notificações.

Componentes da Solução:
Linguagem base: Python

Bibliotecas utilizadas:

mediapipe para detecção de poses e gestos

cv2 (OpenCV) para captura e visualização de vídeo

numpy para tratamento de coordenadas e vetores

Funcionamento:

O sistema captura vídeo de uma camera ou arquivo de simulação

Processa os frames em tempo real usando a Pose Detection do MediaPipe

Identifica sinais de mãos como joinha e punho

Exibe alertas visuais e registra eventos
