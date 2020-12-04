"# Trabalho-CG" 
Matheus Pereira Garbossa

O projeto busta uma automação nos registros de funcinario, eles agora poderão ser reconhecidos por uma camera que identificara seus rostos e o crachá, isso pode tirar a necessidade de um porteiro por exemblo, podendo diminuir ainda mais o contato e aproximação durante essa pandemia.

Nesse projeto foram utilizadas as seguintes libs:

numpy; cv2; face_recognition; os; glob; datetime

O funcinamento da aplicação é bem simples:

Para a identificação de rostos, começa gerando listas de nomes e rostos(estes encontrados na pasta "faces") e então começa a captura de video.

Para a identificação da cor amarela gera uma copia dos frames do video em HSV e cria uma mascara para a cor amarela, então usa esssa mascara no frame gerando um resultado, esse resultado então é convetido para tons de cinza e logo em seguida é aplicada uma binarização. utilizamos o morphologyEx(morph_close) na imagem binaria para remover espaços vazios dentro do objeto e apartir disso geramos uma lista de contornos do objeto.

Voltando a identificação de rostos, a partir dos frames da camera criamos uma copia menor e a transformamos em RGB adquirimos as "locations" e "encodings" a partir desse frame modificado. Agora varendo todos os itens do encodings, para cada um deles verificamos se há um semelhança com os rostos do "faces", caso haja, o nome pertencente a esse rosto sera guardado para ser mostrado na tela depois.

Caso haja a correspondencia de rosto e o nome for guardado iniciaremos então a varedura pela lista de contornos do objeto amarelo, salvando seus dados de x,y,widht,height e calculando a área partir disso, agora caso a área calculada for menor que 3000 nada acontecera, caso o contrario sera escrito no frame da camera que a cor amarela foi detectada e o objeto será contornado, então os dados da data e hora atual serão armazenados em uma string e uma foto sera tirada naquele momento e o nome dela sera a string com as informações armazenadas.

Por final serão unidas as variaveis de "location" e "names" e cada resultante dessa união (top, right, botton, left) será multiplicada pr 4 para retornar ao tamanho original, e a partir dessas resultantes será desenhado no frame da camera um retangulo onde se encontra o rosto e o nome da pessoa registrada a esse rosto, esse é nome que antes foi salvo caso houvesse correspondencia nos rostos.