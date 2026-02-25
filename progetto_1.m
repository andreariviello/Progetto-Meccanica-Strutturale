%% Parte I - Problema della deformazione (Immagine 2D e 3D)

clear; close all; clc

%  - IMMAGINE 2D (francobollo: 1 giugno 1981 - Aereo MB 339 Aermacchi - 200 Lire)
%  - IMMAGINE 3D (cubo colorato)

%% 1.2. Le grandi deformazioni

%% 1.2.1. Funzione piazzamento

% Carico l'immagine del francobollo e la converto in scala di grigi (configurazione di riferimento/Lagrangiana)
img = imread('francobollo.tiff');
img_gray = rgb2gray(img);
img_gray = im2double(img_gray);
[rows, cols] = size(img_gray);

% Definisco la matrice di gradiente di spostamento H
eta = 1.5; % Valore elevato: grande deformazione
H = [0.75 -0.62; 0.27 0.54];

% Calcolo il gradiente del piazzamento (deformation gradient) F secondo la teoria: F = I + ηH
F = eye(2) + eta * H;

% Trovo i nuovi vertici dell'immagine dopo la deformazione: applico la
% trasformazione ai quattro vertici dell'immagine per determinare le
% dimensioni della nuova griglia deformata
corners = [1, cols, cols, 1;
           1, 1, rows, rows];
new_corners = F * corners;
x_min = floor(min(new_corners(1,:)));
x_max = ceil(max(new_corners(1,:)));
y_min = floor(min(new_corners(2,:)));
y_max = ceil(max(new_corners(2,:)));

% Creo la griglia della nuova immagine deformata (configurazione corrente/Euleriana)
[Xq, Yq] = meshgrid(x_min:x_max, y_min:y_max);

% Applico la placement map inversa: ogni pixel della griglia deformata viene riportato nella posizione originale
coords_deform = [Xq(:)'; Yq(:)'];
orig_coords = F \ coords_deform; % per motivi di efficienza computazionale,
% non si inverte direttamente la matrice col comando inv(F) (computazionalmente oneroso)
% ma si risolve invece il sistema lineare associato col comando "\", 
% utilizzando così metodi diretti/iterativi per la risoluzione del sistema lineare.
%
% Verrà utilizzato questo approccio in tutto il codice. 
% Alternativamente se si vuole invertire esplicitamente la matrice si procede, 
% per esempio, come segue:
%
% invF = inv(F);
% orig_coords = invF * coords_deform;

Xorig = reshape(orig_coords(1,:), size(Xq));
Yorig = reshape(orig_coords(2,:), size(Yq));

% L'interpolazione bilineare dei valori di grigio (interp2) calcola il valore 
% deformato dell'immagine, mapping inverso: da punti nella configurazione 
% corrente (Euleriana) a corrispondenti punti di configurazione di riferimento.
% Valori fuori dominio sono saturati a 0.97 per simulare lo sfondo chiaro.
img_transform_full = interp2(1:cols, 1:rows, img_gray, Xorig, Yorig, 'linear', 0.97);

% Visualizzo a confronto l'immagine originale e quella deformata
% Figure 1.2: Effetto delle grandi deformazioni - caso 2D

figure(1);

subplot(1,2,1);
imshow(img_gray, []);
title('Francobollo Originale');
xlabel('X'); ylabel('Y'); grid on;
axis on; xticks(0:100:cols); yticks(0:100:rows);

subplot(1,2,2);
imshow(img_transform_full, 'XData', [x_min x_max], 'YData', [y_min y_max]);
title(['Francobollo - Grande Deformazione (\eta = ' num2str(eta) ')']);
xlabel('x'); ylabel('y'); grid on;
axis on; xticks(-300:150:1500); yticks(0:150:900);

% CAMPO VETTORIALE DI SPOSTAMENTO (CASO 2D):

% Scelgo un passo ampio per ridurre il numero di vettori e facilizzare la visualizzazione
step = 50;
% Genero una griglia regolare sulla configurazione originale, includendo anche l'ultimo bordo
[X_grid, Y_grid] = meshgrid([1:step:cols, cols], [1:step:rows, rows]);
% Dispongo le coordinate originali come vettori 2xN per applicare la trasformazione
coords_X = [X_grid(:)'; Y_grid(:)'];
% Applico il gradiente di piazzamento F per ottenere le nuove coordinate deformate
coords_x = F * coords_X;
% Estraggo le coordinate di partenza (punti originali)
X_start = coords_X(1, :);
Y_start = coords_X(2, :);

% Estraggo le coordinate di arrivo (punti deformati)
X_end = coords_x(1, :);
Y_end = coords_x(2, :);

% Calcolo i vettori spostamento
Sx_sample = X_end - X_start;
Sy_sample = Y_end - Y_start;

% Calcolo la norma (magnitudine) di ogni vettore per la visualizzazione colorata
mag = sqrt(Sx_sample.^2 + Sy_sample.^2);
cmap = jet(256);
mag_vec = mag(:);
mag_range = max(mag_vec) - min(mag_vec);
color_idx = round(1 + (mag_vec - min(mag_vec)) / mag_range * 255);
color_idx = max(min(color_idx,256),1);

% Calcolo i vertici deformati dell'immagine
corners = [1, cols, cols, 1; 1, 1, rows, rows];
new_corners = F * corners;

% Fisso i limiti del plot includendo anche i vertici dell'immagine deformata
all_X_deformed = [X_end, new_corners(1,:)];
all_Y_deformed = [Y_end, new_corners(2,:)];
x_min_plot = floor(min(all_X_deformed));
x_max_plot = ceil(max(all_X_deformed));
y_min_plot = floor(min(all_Y_deformed));
y_max_plot = ceil(max(all_Y_deformed));

% Figure 1.3: Campo di spostamento - caso 2D

figure(2)

hold on;
for k = 1:length(X_start)
    col = cmap(color_idx(k), :);
    quiver(X_start(k), Y_start(k), Sx_sample(k),Sy_sample(k), 0, ...
           'Color', col, 'LineWidth', 1.5, 'MaxHeadSize', 5);
end
hold off;

% Impostazioni per una visualizzazione ottimale del grafico
set(gca, 'YDir', 'reverse');
xlabel('x');
ylabel('y');
title(['Francobollo - Campo Vettoriale di Spostamento (\eta = ' num2str(eta) ')']);
colormap(cmap);
cbar = colorbar;
cbar.Label.String = 'Magnitudine spostamento';
grid on;
axis([x_min_plot x_max_plot y_min_plot y_max_plot]);

% Ogni vettore viene disegnato con colore variabile (colormap jet) in base alla sua magnitudine,
% facilitando così l'identificazione visiva delle zone di maggiore o minore deformazione.

% Commento teorico:
% Il campo vettoriale di spostamento deriva dalla funzione di piazzamento che mappa la
% configurazione di riferimento X sulla configurazione attuale x. Il gradiente F descrive
% le variazioni locali nello spazio materiale e determina la direzione e la magnitudine
% degli spostamenti. Il campo vettoriale di spostamento s(X) rappresenta quanto e in che
% direzione ogni punto materiale si sposta, ed è calcolato come u(X) = χ - X = (F - I)X = ηHX, 
% funzione lineare nello spazio.
% La sua visualizzazione consente di interpretare il moto e la deformazione locale.

%% 1.2.2. Decomposizione polare

% La decomposizione polare scompone il deformation gradient F nella matrice di stiramento U
% e nella matrice di rotazione R. Questa separazione chiarisce la natura della deformazione e del movimento rigido.

U = sqrtm(F' * F);

cornersU = U * corners;
x_minU = floor(min(cornersU(1,:)));
x_maxU = ceil(max(cornersU(1,:)));
y_minU = floor(min(cornersU(2,:)));
y_maxU = ceil(max(cornersU(2,:)));

[XqU, YqU] = meshgrid(x_minU:x_maxU, y_minU:y_maxU);
invmapU = U \ [XqU(:)'; YqU(:)'];
XorigU = reshape(invmapU(1,:), size(XqU));
YorigU = reshape(invmapU(2,:), size(YqU));

img_U = interp2(1:cols, 1:rows, img_gray, XorigU, YorigU, 'linear', 0.97);

R = F / U;   % Risolve R*U = F
cornersR = R * corners; % analogamente a quanto detto prima, anche qui è possibile
% non invertire esplicitamente la matrice U e risolvere il sistema lineare
% associato tramite metodi diretti attraverso il comando MATLAB \
% (numericamente più efficiente)
x_minR = floor(min(cornersR(1,:)));
x_maxR = ceil(max(cornersR(1,:)));
y_minR = floor(min(cornersR(2,:)));
y_maxR = ceil(max(cornersR(2,:)));

[XqR, YqR] = meshgrid(x_minR:x_maxR, y_minR:y_maxR);
invmapR = R \ [XqR(:)'; YqR(:)'];
XorigR = reshape(invmapR(1,:), size(XqR));
YorigR = reshape(invmapR(2,:), size(YqR));

img_R = interp2(1:cols, 1:rows, img_gray, XorigR, YorigR, 'linear', 0.97);

% Figure 1.4: Decomposizione polare destra (R ed U)

figure(3)

subplot(1,3,1);
imshow(img_gray, []);
title('Francobollo Originale');
xlabel('X'); ylabel('Y'); grid on;
axis on; xticks(0:100:cols); yticks(0:100:rows);

subplot(1,3,2);
imshow(img_U, 'XData', [x_minU x_maxU], 'YData', [y_minU y_maxU]);
title('Francobollo - solo U');
xlabel('x'); ylabel('y'); grid on;
axis on; xticks(-150:150:1500); yticks(-150:150:750);

subplot(1,3,3);
imshow(img_R, 'XData', [x_minR x_maxR], 'YData', [y_minR y_maxR]);
title('Francobollo - solo R');
xlabel('x'); ylabel('y'); grid on;
axis on; xticks(-200:100:900); yticks(0:100:600);

%% 1.2.3. Rotazione pura

gamma = deg2rad(30);
F_rot = [cos(gamma), -sin(gamma); sin(gamma), cos(gamma)];

corners_rot = F_rot * corners;
x_minRot = floor(min(corners_rot(1,:)));
x_maxRot = ceil(max(corners_rot(1,:)));
y_minRot = floor(min(corners_rot(2,:))); 
y_maxRot = ceil(max(corners_rot(2,:)));
[XqRot, YqRot] = meshgrid(x_minRot:x_maxRot, y_minRot:y_maxRot);
invmapRot = F_rot \ [XqRot(:)'; YqRot(:)'];
XorigRot = reshape(invmapRot(1,:), size(XqRot)); 
YorigRot = reshape(invmapRot(2,:), size(YqRot));
img_rot = interp2(1:cols, 1:rows, img_gray, XorigRot, YorigRot, 'linear', 0.97);

% Figure 1.5: Rotazione pura - caso 2D (γ = 30°)

figure(4);

subplot(1,2,1);
imshow(img_gray, []);
title('Francobollo Originale');
xlabel('X'); ylabel('Y'); grid on;
axis on; xticks(0:100:cols); yticks(0:100:rows);

subplot(1,2,2);
imshow(img_rot, 'XData', [x_minRot x_maxRot], 'YData', [y_minRot y_maxRot]);
title('Francobollo - Rotazione Pura (\gamma = 30°)');
xlabel('x'); ylabel('y'); grid on;
axis on; xticks(-200:100:600); yticks(0:100:600);

%% 1.2.4. Misurare la deformazione

% Calcolo del tensore di deformazione finita di Green-Lagrange (E)
% Definito come E = 1/2 * (F' * F - I), misura la variazione di lunghezza al quadrato tra
% la configurazione deformata e quella di riferimento (configurazione Lagrangiana).
E = 1/2 * (F' * F - eye(2));
% Definisce una fibra materiale unitaria T nella configurazione di riferimento
% Qui ad esempio una fibra verticale lungo l'asse Y
T = [0; 1];

% Calcola l'allungamento di Green-Lagrange nella direzione della fibra T.
% Questa è la componente di deformazione finita secondo E:
epsilon_diretta = T' * E * T; % risultante deformazione di Green-Lagrange nella direzione di T

% La deformazione nominale (o ingegneristica), che corrisponde all'allungamento relativo
% (l - L) / L, si calcola a partire da epsilon_diretta:
epsilon_nominale = sqrt(1 + 2 * epsilon_diretta) - 1;

% Con i valori di grande deformazione (ad esempio epsilon_diretta = 1.5705):
% Il calcolo fornisce: epsilon_nominale = sqrt(1 + 2 * 1.5705) - 1 = 1.0349
% Questo corrisponde a una deformazione relativa del 103.49% lungo la fibra T.

%% 1.3. Le piccole deformazioni

%% 1.3.1. Teoria linearizzata

eta_small = 0.05; % valore molto contenuto del parametro η
F_small = eye(2) + eta_small * H;

new_corners_small = F_small * corners;
x_minS = floor(min(new_corners_small(1,:)));
x_maxS = ceil(max(new_corners_small(1,:)));
y_minS = floor(min(new_corners_small(2,:)));
y_maxS = ceil(max(new_corners_small(2,:)));

[XqS, YqS] = meshgrid(x_minS:x_maxS, y_minS:y_maxS);
invmapS = F_small \ [XqS(:)'; YqS(:)'];
XorigS = reshape(invmapS(1,:), size(XqS));
YorigS = reshape(invmapS(2,:), size(YqS));

img_small = interp2(1:cols, 1:rows, img_gray, XorigS, YorigS, 'linear', 0.97);

% Figure 1.6: Effetto delle piccole deformazioni

figure(5);

subplot(1,2,1);
imshow(img_gray, []);
title('Francobollo Originale');
xlabel('X'); ylabel('Y'); grid on;
axis on; xticks(0:100:cols); yticks(0:100:rows);

subplot(1,2,2);
imshow(img_small, 'XData', [x_minS x_maxS], 'YData', [y_minS y_maxS]);
title(['Francobollo - Piccola Deformazione (\eta = ' num2str(eta_small) ')']);
xlabel('x'); ylabel('y'); grid on;
axis on; xticks(0:100:cols); yticks(0:100:rows);

%% 1.3.2. Decomposizione additiva

% La decomposizione additiva (nel caso di piccole deformazioni) scompone 
% il gradiente di spostamento H_add in una parte simmetrica (ε, deformazione lineare) 
% e una antisimmetrica/emisimmetrica (W, rotazione infinitesimale),
    
% Calcola il gradiente di spostamento per piccole deformazioni
H_add = eta_small * H;

% Decomposizione in parte simmetrica (ε) e antisimmetrica (W)
epsilon = 0.5 * (H_add + H_add');  % Parte simmetrica: rappresenta la deformazione pura (ε, strain tensor)
W = 0.5 * (H_add - H_add');        % Parte antisimmetrica: rappresenta la rotazione infinitesimale (W, spin tensor)

% Ricostruzione delle matrici di trasformazione associate a ε e W
F_epsilon = eye(2) + epsilon;  % Trasformazione pura di strain
F_W = eye(2) + W;              % Trasformazione pura di rotazione

corners_epsilon = F_epsilon * corners;  % Nuova posizione dei vertici dopo ε
x_minE = floor(min(corners_epsilon(1,:)));
x_maxE = ceil(max(corners_epsilon(1,:)));
y_minE = floor(min(corners_epsilon(2,:)));
y_maxE = ceil(max(corners_epsilon(2,:)));

% Crea la griglia della nuova immagine deformata da ε   
[XqE, YqE] = meshgrid(x_minE:x_maxE, y_minE:y_maxE);

% Applica la placement map inversa: mappatura dalla configurazione deformata a quella originale
invmapE = F_epsilon \ [XqE(:)'; YqE(:)'];
XorigE = reshape(invmapE(1,:), size(XqE));
YorigE = reshape(invmapE(2,:), size(YqE));

% Interpolazione lineare dei valori di grigio sull'immagine originale
img_epsilon = interp2(1:cols, 1:rows, img_gray, XorigE, YorigE, 'linear', 0.97);

% Ripeto la medesima operazione per la rotazione pura W
corners_W = F_W * corners;
x_minW = floor(min(corners_W(1,:)));
x_maxW = ceil(max(corners_W(1,:)));
y_minW = floor(min(corners_W(2,:)));
y_maxW = ceil(max(corners_W(2,:)));
[XqW, YqW] = meshgrid(x_minW:x_maxW, y_minW:y_maxW);
invmapW = F_W \ [XqW(:)'; YqW(:)'];
XorigW = reshape(invmapW(1,:), size(XqW));
YorigW = reshape(invmapW(2,:), size(YqW));
img_W = interp2(1:cols, 1:rows, img_gray, XorigW, YorigW, 'linear', 0.97);

% Visualizzazione comparata delle due componenti
% Figure 1.7: Decomposizione additiva (ε e W)

figure(6);

subplot(1,3,1);
imshow(img_gray, []);
title('Francobollo Originale');
xlabel('X'); ylabel('Y'); grid on;
axis on; xticks(0:100:cols); yticks(0:100:rows);

subplot(1,3,2);
imshow(img_epsilon, 'XData', [x_minE x_maxE], 'YData', [y_minE y_maxE]);
title('Francobollo - solo \epsilon');
xlabel('x'); ylabel('y'); grid on;
axis on; xticks(0:100:cols); yticks(0:100:rows);

subplot(1,3,3);
imshow(img_W, 'XData', [x_minW x_maxW], 'YData', [y_minW y_maxW]);
title('Francobollo - solo W');
xlabel('x'); ylabel('y'); grid on;
axis on; xticks(0:100:cols); yticks(0:100:rows);

% Commento teorico:
% La matrice H_add è il gradiente di spostamento scalato per simulare piccole deformazioni.
% La decomposizione additiva di H_add in ε (parte simmetrica) e W (parte antisimmetrica)
% riflette la separazione tra deformazione pura e rotazione infinitesimale.
% L'uso della placement map inversa e dell'interpolazione 2D permette di ricostruire
% visivamente l'effetto isolato della deformazione e della rotazione sulla configurazione originale.

%% 1.3.3. Misurare la deformazione

% Calcolo del tensore di deformazione finita di Green-Lagrange (E)
% Definito come E = 1/2 * (F' * F - I), misura la variazione di lunghezza al quadrato tra
% la configurazione deformata e quella di riferimento (configurazione Lagrangiana).
E_small = 1/2 * (F_small' * F_small - eye(2));

% Definisce una fibra materiale unitaria T nella configurazione di riferimento
% Qui ad esempio una fibra verticale lungo l'asse Y
T = [0; 1];

% Calcola la deformazione diretta della fibra T, valutando l'allungamento relativo
% secondo la forma quadratica epsilon_diretta = T' * E * T
epsilon_diretta_small = T' * E_small * T; % pari a circa 0.0278
% Nel caso di piccole deformazioni, come spiegato nella teoria, la misura di 
% deformazione diretta (epsilon_diretta_small) è molto vicina (approssima)
% la misura di deformazione nominale/ingegneristica (epsilon_nominale_small).

% La misura di deformazione nominale, espressa come allungamento relativo:
epsilon_nominale_small = sqrt(1 + 2 * T' * E_small * T) - 1;
% Se epsilon_diretta_small = 0.0278, allora epsilon_nominale_small = 0.0275.
% Questo corrisponde a una deformazione relativa del 2.75% lungo la fibra T.

% MISURA SCORRIMENTO ANGOLARE:

% Definisco due fibre materiali iniziali mutuamente ortogonali:
T_alpha = [1; 0];
T_beta = [0; -1];

% Calcolo lo scorrimento angolare con l'ipotesi di piccole deformazioni:
gamma_alphabeta_rad = 2 * T_alpha' * epsilon * T_beta;
gamma_alphabeta_deg = rad2deg(gamma_alphabeta_rad);
% Lo scorrimento angolare risulta pari a 1.0027°.

cos_theta = sin(gamma_alphabeta_rad); 
theta_finale = acosd(cos_theta);
% L'angolo finale tra le fibre sarà dunque equivalente a 88.9973°.

%% 1.4. Il caso tridimensionale

clear; clc;

% Creo una griglia regolare di punti nello spazio 3D (configurazione di riferimento/Lagrangiana)
n = 90;
[x, y, z] = meshgrid(linspace(1, n, n), linspace(1, n, n), linspace(1, n, n));

% Definisco una funzione scalare sui punti del cubo
V = x + y + z; % funzione "cubo colorato"

% Definisco i vertici del cubo 
verts = [1 1 1; n 1 1; n n 1; 1 n 1; 1 1 n; n 1 n; n n n; 1 n n]';

% Definisco gli spigoli del cubo come coppie di indici nei vertici
edges = [1 2; 2 3; 3 4; 4 1;  % base inferiore
         5 6; 6 7; 7 8; 8 5;  % base superiore
         1 5; 2 6; 3 7; 4 8]; % verticali

% Figure 1.8: Effetto delle grandi deformazioni - caso 3D

figure(7);

% Plot cubo originale:
subplot(1,2,1);
% Visualizzo tutti i punti del cubo come nuvola colorata (scatter3)
scatter3(x(:), y(:), z(:), 10, V(:), 'filled'); hold on;
% Disegno i bordi neri del cubo (spigoli) per una migliore visualizzazione
for i = 1:size(edges,1)
    plot3(verts(1,edges(i,:)), verts(2,edges(i,:)), verts(3,edges(i,:)), 'k', 'LineWidth', 4);
end
colormap(cool);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Cubo 3D Originale');
axis equal tight vis3d;
axis([0 180 0 180 -10 130]);
grid on; xticks(0:30:180); yticks(0:30:180); zticks(-10:30:130);
view(3); hold off;

% GRANDI DEFORMAZIONI (CASO 3D):

% Definisco la matrice di gradiente di spostamento H (3x3)
% H rappresenta il gradiente del campo di spostamento tridimensionale:
% H(i,j) = derivata della componente i-esima dello spostamento rispetto alla coordinata j
% In questo esempio, H è scelta
%  direttamente per simulare una deformazione generica
% Non deriva da un campo di spostamento esplicito, ma è una scelta numerica/didattica
eta = 4.4; % Intensità della deformazione (valore elevato per evidenziare l'effetto)
H = [0.1 0.05 0.02;
     0.04 0.15 0.03;
     0.01 -0.02 0.08];

% Calcolo il deformation gradient F = I + ηH
F_def = eye(3) + eta * H;

% Applico la deformazione a tutti i punti del cubo (configurazione deformata/Euleriana)
coords_def = F_def * [x(:)'; y(:)'; z(:)'];
Xdef = coords_def(1,:)';
Ydef = coords_def(2,:)';
Zdef = coords_def(3,:)';

% Applico la stessa deformazione ai vertici del cubo per disegnare i bordi
% in nero per una migliore visualizzazione (spigoli)
verts_def = F_def * verts;

% Plot cubo deformato:
subplot(1,2,2);
scatter3(Xdef, Ydef, Zdef, 10, V(:), 'filled'); hold on;
for i = 1:size(edges,1)
    plot3(verts_def(1,edges(i,:)), verts_def(2,edges(i,:)), verts_def(3,edges(i,:)), 'k', 'LineWidth', 4);
end
colormap(cool);
xlabel('x'); ylabel('y'); zlabel('z');
title(['Cubo 3D - Grande Deformazione (\eta = ' num2str(eta) ')']);
axis tight equal vis3d;
axis([0 180 0 180 -10 130]);
grid on; xticks(0:30:180); yticks(0:30:180); zticks(-10:30:130);
view(3); hold off;

% CAMPO VETTORIALE DI SPOSTAMENTO (CASO 3D):

n = 7;
[x, y, z] = meshgrid(linspace(1,80,n), linspace(1,80,n), linspace(1,80,n));
coords = [x(:)'; y(:)'; z(:)']; 

% Calcolo spostamenti
s = eta * H * coords;
s_x = reshape(s(1,:), size(x));
s_y = reshape(s(2,:), size(y));
s_z = reshape(s(3,:), size(z));

% Colore proporzionale alla magnitudine (norma euclidea) dello spostamento:

% Calcolo la magnitudine di ciascun vettore
magnitude = sqrt(s_x(:).^2 + s_y(:).^2 + s_z(:).^2);

% Normalizzazione per la colormap:
mag_norm = (magnitude - min(magnitude)) / (max(magnitude) - min(magnitude));
cmap = jet(256);
color_idx = round(1 + mag_norm * 255);
color_idx(color_idx < 1) = 1;
color_idx(color_idx > 256) = 256;

% Figure 1.9: Campo di spostamento - caso 3D

figure(8);

hold on;
for k = 1:numel(s_x)
    col = cmap(color_idx(k), :);
    quiver3(x(k), y(k), z(k), s_x(k), s_y(k), s_z(k), 0, ...
            'Color', col, 'LineWidth', 1.3);
end
hold off;
xlabel('X'); ylabel('Y'); zlabel('Z');
title(['Cubo 3D - Campo vettoriale di spostamento (\eta = ' num2str(eta) ')']);
colormap(cmap); view([135 30]);
cbar = colorbar;
cbar.Label.String = 'Magnitudine spostamento';
axis equal tight vis3d;
set(gca, 'XTickLabel', []);
set(gca, 'YTickLabel', []);
set(gca, 'ZTickLabel', []);
grid on;
view(3); hold off;

% Commento teorico:
% Anche nel caso tridimensionale il campo di spostamento s(X) = (F - I)*X 
% deriva dalla funzione di piazzamento e mostra il moto locale di ogni 
% punto materiale a seguito della deformazione totale definita da F.

% ROTAZIONE PURA (CASO 3D):

% Definisco la matrice di rotazione attorno all'asse Z
gamma = deg2rad(45);
Rz = [cos(gamma), -sin(gamma), 0;
      sin(gamma),  cos(gamma), 0;
      0,           0,          1];

n = 90;
[x, y, z] = meshgrid(linspace(1, n, n), linspace(1, n, n), linspace(1, n, n));

% Applico la rotazione a tutti i punti del cubo
coords_rot = Rz * [x(:)'; y(:)'; z(:)'];
Xrot = coords_rot(1,:)';
Yrot = coords_rot(2,:)';
Zrot = coords_rot(3,:)';

% Applico la stessa rotazione ai vertici del cubo per disegnare i bordi
% in nero per una migliore visualizzazione (spigoli)
verts_rot = Rz * verts;

% Figure 1.10: Rotazione pura - caso 3D (γ = 45°)

figure(9);

subplot(1,2,1);
% Visualizzo tutti i punti del cubo come nuvola colorata (scatter3)
scatter3(x(:), y(:), z(:), 10, V(:), 'filled'); hold on;
% Disegno i bordi neri del cubo per una migliore visualizzazione (spigoli)
for i = 1:size(edges,1)
    plot3(verts(1,edges(i,:)), verts(2,edges(i,:)), verts(3,edges(i,:)), 'k', 'LineWidth', 4);
end
colormap(cool);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Cubo 3D Originale');
axis equal tight vis3d;
axis([-1 139 -1 139 -1 91]);
grid on; xticks(0:30:140); yticks(0:30:130); zticks(0:30:90);
view(3); hold off;

subplot(1,2,2);
scatter3(Xrot, Yrot, Zrot, 10, V(:), 'filled'); hold on;
for i = 1:size(edges,1)
    plot3(verts_rot(1,edges(i,:)), verts_rot(2,edges(i,:)), verts_rot(3,edges(i,:)), 'k', 'LineWidth', 4);
end
colormap(cool);
xlabel('x'); ylabel('y'); zlabel('z');
title('Cubo 3D - Rotazione Pura (\gamma = 45°)');
axis equal tight vis3d;
axis([-70 70 -1 139 -1 91]);
grid on; xticks(-60:30:60); yticks(0:30:130); zticks(0:30:90);
view(3); hold off;