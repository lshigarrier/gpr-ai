"""
Objectif du code :
Ce script traite une arborescence de dossiers contenant des images PNG (B-scans) et y superpose
des horizons géologiques à partir de fichiers CSV (layers). Il convertit les épaisseurs des couches
(en cm) en temps de parcours (ns) en utilisant un fichier de vitesses, puis dessine ces couches en
couleur sur les B-scans convertis en RGB. Les images générées reproduisent l'arborescence d'entrée.
Une légende est générée à la racine du dossier de sortie.

Entrées :
- Dossier d'entrée (contenant les sous-dossiers et les PNG).
- Fichier velocities.csv (colonnes: layer_id, velocity).
- Fichiers layers_xxx.csv (colonnes: coord_x, coord_y, layer_1, ..., layer_n).

Sorties :
- Un dossier de sortie avec l'arborescence conservée contenant les images annotées.
- Une image legend.png à la racine du dossier de sortie.

Exemple de commande :
python add_gt_to_png.py config_gt_to_png
"""

import csv
from scipy.spatial import KDTree
from pathlib import Path
from PIL import Image, ImageDraw, PngImagePlugin

from utils import get_conf


def load_velocities(velocities_path: Path) -> dict:
    velocities = {}
    with velocities_path.open(mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            velocities[row['layer_id']] = float(row['velocity'])
    return velocities


def load_layers(layers_paths: list, velocities: dict) -> dict:
    layers_data = {}

    for path in layers_paths:
        layer_name = path.stem
        coords = []
        times = []

        with path.open(mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            layer_cols = [h for h in headers if h.startswith("layer_")]

            for row in reader:
                x = float(row['coord_x'])
                y = float(row['coord_y'])
                coords.append([x, y])

                point_times = []
                cumul_time = 0.0
                for col in layer_cols:
                    thickness_cm = float(row[col])
                    vel = velocities[col]
                    time_ns = 2 * thickness_cm / vel
                    cumul_time += time_ns
                    point_times.append(cumul_time)

                times.append(point_times)

        tree = KDTree(coords)
        layers_data[layer_name] = {
            "tree": tree,
            "times": times,
            "layer_cols": layer_cols
        }

    return layers_data


def generate_colors(n: int) -> dict:
    colors = {}
    # Génération de couleurs réparties sur le cercle chromatique pour maximiser la différence
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        # Conversion simple HSV vers RGB (saturation=1, value=1)
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        p = 0
        q = int(255 * (1 - f))
        t = int(255 * (1 - (1 - f)))
        v = 255
        if h_i % 6 == 0:
            r, g, b = v, t, p
        elif h_i % 6 == 1:
            r, g, b = q, v, p
        elif h_i % 6 == 2:
            r, g, b = p, v, t
        elif h_i % 6 == 3:
            r, g, b = p, q, v
        elif h_i % 6 == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        colors[i] = (r, g, b)
    return colors


def process_image(img_path: Path, out_path: Path, layers_data: dict, colors_map: dict,
                  margin: int, thickness: int, alpha:float):
    # Création du dossier parent de l'image de sortie
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path)
    rgb_img = img.convert("RGB")
    pixels = rgb_img.load()

    # Lecture des métadonnées
    meta = img.text
    gps_x_start = float(meta["GPS_X_START"])
    gps_y_start = float(meta["GPS_Y_START"])
    gps_x_end = float(meta["GPS_X_END"])
    gps_y_end = float(meta["GPS_Y_END"])
    time_step_ns = float(meta["TIME_STEP_NS"])

    width, height = rgb_img.size
    bscan_w = width - 2 * margin
    bscan_h = height - 2 * margin

    half_thick = thickness // 2

    # Accumulateur pour stocker { (x, y): [somme_R, somme_G, somme_B, nombre_de_couches] }
    color_accumulator = {}

    # Parcours des A-scans (colonnes du B-scan sans marges)
    for i in range(bscan_w):
        # Interpolation spatiale
        fraction = i / max(1, bscan_w - 1)
        curr_x = gps_x_start + fraction * (gps_x_end - gps_x_start)
        curr_y = gps_y_start + fraction * (gps_y_end - gps_y_start)

        for layer_name, data in layers_data.items():
            tree = data["tree"]
            times_list = data["times"]

            # Recherche du point le plus proche
            _, idx = tree.query([curr_x, curr_y])
            point_times = times_list[idx]

            cr, cg, cb = colors_map[layer_name]

            for t_ns in point_times:
                # Calcul de l'index du pixel en y
                y_pixel = int(round(t_ns / time_step_ns))

                # Détermination des bornes de la bande autour du centre
                y_start = y_pixel - half_thick
                y_end = y_start + thickness

                for dy in range(y_start, y_end):
                    # Vérification que le point reste dans l'image (sans recouvrir les marges)
                    if 0 <= dy < bscan_h:
                        final_x = i + margin
                        final_y = dy + margin

                        # Ajout à l'accumulateur au lieu de modifier le pixel immédiatement
                        if (final_x, final_y) not in color_accumulator:
                            color_accumulator[(final_x, final_y)] = [0, 0, 0, 0]

                        color_accumulator[(final_x, final_y)][0] += cr
                        color_accumulator[(final_x, final_y)][1] += cg
                        color_accumulator[(final_x, final_y)][2] += cb
                        color_accumulator[(final_x, final_y)][3] += 1

    # Application des couleurs accumulées sur l'image
    for (x, y), (r_sum, g_sum, b_sum, count) in color_accumulator.items():
        pr, pg, pb = pixels[x, y]

        # Moyenne des couleurs des layers pour ce pixel
        avg_cr = r_sum // count
        avg_cg = g_sum // count
        avg_cb = b_sum // count

        # Mélange final : B-scan d'origine pondéré par (1 - alpha) + moyenne des layers pondérée par alpha
        new_r = int(pr * (1 - alpha) + avg_cr * alpha)
        new_g = int(pg * (1 - alpha) + avg_cg * alpha)
        new_b = int(pb * (1 - alpha) + avg_cb * alpha)

        pixels[x, y] = (new_r, new_g, new_b)

    # Préservation des métadonnées d'origine
    png_info = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        png_info.add_text(k, v)

    rgb_img.save(out_path, "PNG", pnginfo=png_info)


def create_legend(output_dir: Path, colors_map: dict):
    square_size = 30
    padding = 10
    width = 300
    height = (square_size + padding) * len(colors_map) + padding

    legend_img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(legend_img)

    for idx, (layer_name, color) in enumerate(colors_map.items()):
        y0 = padding + idx * (square_size + padding)
        x0 = padding
        y1 = y0 + square_size
        x1 = x0 + square_size

        draw.rectangle([x0, y0, x1, y1], fill=color)
        draw.text((x1 + padding, y0 + square_size // 4), layer_name, fill="black")

    legend_img.save(output_dir / "legend.png", "PNG")


def main():
    conf = get_conf(verbose=False)
    input_dir = Path(conf.input_dir)
    output_dir = Path(conf.output_dir)
    velocities_path = Path(conf.velocities_path)
    layers_paths = [Path(p) for p in conf.layers_paths]

    # Chargement et préparation des données
    velocities = load_velocities(velocities_path)
    layers_data = load_layers(layers_paths, velocities)

    # Attribution des couleurs
    raw_colors = generate_colors(len(layers_data))
    colors_map = {name: raw_colors[i] for i, name in enumerate(layers_data.keys())}

    # Traitement des images PNG
    print("Start processing images...")
    for img_path in input_dir.rglob("*.png"):
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path

        process_image(img_path, out_path, layers_data, colors_map, conf.margin, conf.thickness, conf.alpha)

    # Génération de la légende à la racine du dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    create_legend(output_dir, colors_map)
    print("Processing completed.")


if __name__ == "__main__":
    main()
