"""
Objectif du code :
Ce script traite une arborescence de dossiers contenant des images PNG (B-scans) et y superpose
des horizons géologiques à partir de fichiers CSV. L'application est multi-planches : chaque image
est d'abord géo-localisée par trace (A-scan) pour déterminer à quelle emprise de planche elle appartient.
Ensuite, les épaisseurs des couches de vérité terrain spécifiques à cette planche sont converties
en temps de parcours (ns) à l'aide d'un fichier de vitesses propre à la planche, puis dessinées
en couleur sur les B-scans. Une légende unique globale regroupant toutes les couches est générée.

Entrées :
- Fichiers de configuration définissant les répertoires d'entrée/sortie.
- Dictionnaires de chemins pour chaque planche (footprints, velocities, layers).

Sorties :
- Un dossier de sortie avec l'arborescence conservée contenant les images annotées.
- Une image legend.png unique à la racine du dossier de sortie.

Exemple de commande :
python add_gt_to_png.py config_gt_to_png
"""

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import KDTree
from matplotlib.path import Path as MplPath
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from utils import get_conf


def load_footprints(footprint_paths: dict) -> dict:
    """
    Charge les fichiers CSV d'emprises pour chaque planche et convertit
    les coordonnées EPSG:4326 (lat/lon) vers EPSG:32631 (mètres).
    Renvoie un dictionnaire de polygones matplotlib.path.Path pour un test d'inclusion vectorisé.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    footprints = {}

    for planche_id, csv_path in footprint_paths.items():
        df = pd.read_csv(csv_path)
        xs, ys = transformer.transform(df['longitude'].values, df['latitude'].values)
        vertices = np.column_stack((xs, ys))
        footprints[planche_id] = MplPath(vertices)

    return footprints


def load_velocities(velocities_paths: dict) -> dict:
    """
    Charge les vitesses pour chaque planche.
    Renvoie un dictionnaire imbriqué : {planche_id: {layer_id: velocity, ...}, ...}
    """
    velocities = {}
    for planche_id, csv_path in velocities_paths.items():
        df = pd.read_csv(csv_path)
        velocities[planche_id] = dict(zip(df['layer_id'], df['velocity']))
    return velocities


def load_layers(layers_paths: dict, velocities: dict) -> tuple:
    """
    Charge les fichiers de couches, projette les points et construit les arbres KDTrees
    par planche. Les temps de parcours sont calculés en amont de manière vectorisée.

    Retourne :
    - layers_data : structure contenant les KDTrees et les temps précalculés par planche.
    - all_layer_names : liste de tous les noms de couches uniques (pour la légende).
    """
    layers_data = {}
    all_layer_names = []
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)

    for planche_id, paths_list in layers_paths.items():
        layers_data[planche_id] = {}
        planche_vels = velocities[planche_id]

        for path in paths_list:
            layer_name = path.stem
            if layer_name not in all_layer_names:
                all_layer_names.append(layer_name)

            df = pd.read_csv(path)
            xs, ys = transformer.transform(df['longitude'].values, df['latitude'].values)
            coords = np.column_stack((xs, ys))
            tree = KDTree(coords)

            layer_cols = [c for c in df.columns if c.startswith("layer_")]
            times = np.zeros((len(df), len(layer_cols)))

            # Calcul cumulé des temps par point de la vérité terrain
            cumul_time = np.zeros(len(df))
            for i, col in enumerate(layer_cols):
                thickness_cm = df[col].values
                vel = planche_vels[col]
                time_ns = 2 * thickness_cm / vel
                cumul_time += time_ns
                times[:, i] = cumul_time

            layers_data[planche_id][layer_name] = {
                "tree": tree,
                "times": times
            }

    return layers_data, all_layer_names


def generate_colors(n: int) -> dict:
    """Génère un dictionnaire de couleurs uniques et bien distinctes."""
    colors = {}
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        p = 0
        q = int(255 * (1 - f))
        t = int(255 * (1 - (1 - f)))
        v = 255

        if h_i % 6 == 0:   r, g, b = v, t, p
        elif h_i % 6 == 1: r, g, b = q, v, p
        elif h_i % 6 == 2: r, g, b = p, v, t
        elif h_i % 6 == 3: r, g, b = p, q, v
        elif h_i % 6 == 4: r, g, b = t, p, v
        else:              r, g, b = v, p, q

        colors[i] = (r, g, b)
    return colors


def process_image(img_path: Path, out_path: Path, footprints: dict, layers_data: dict,
                  colors_map: dict, margin: int, thickness: int, alpha: float):
    """
    Traite une image PNG en y incrustant les couleurs de vérité terrain.
    Détermine d'abord pour chaque A-scan sa planche d'appartenance puis trouve
    le point le plus proche dans les KDTrees associés de façon vectorisée.
    """
    img = Image.open(img_path)
    rgb_img = img.convert("RGB")
    pixels = rgb_img.load()

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

    # Interpolation vectorisée des coordonnées de tous les A-scans
    fractions = np.linspace(0, 1, bscan_w)
    curr_xs = gps_x_start + fractions * (gps_x_end - gps_x_start)
    curr_ys = gps_y_start + fractions * (gps_y_end - gps_y_start)
    a_scan_coords = np.column_stack((curr_xs, curr_ys))

    # Détermination de l'appartenance de chaque A-scan à une emprise (planche)
    assigned_planche = np.full(bscan_w, "", dtype=object)
    for planche_id, poly in footprints.items():
        in_poly = poly.contains_points(a_scan_coords)
        mask_to_assign = in_poly & (assigned_planche == "")
        assigned_planche[mask_to_assign] = planche_id

    # Dictionnaire d'accumulation de la couleur: {(x, y): [R_sum, G_sum, B_sum, count]}
    color_accumulator = {}

    # Itération par planche trouvée dans cette image (pour requête KDTree groupée)
    unique_planches = np.unique(assigned_planche)
    for planche_id in unique_planches:
        if planche_id == "":
            continue  # A-scans en dehors de toute emprise

        # Indices des A-scans concernés par cette planche
        indices = np.where(assigned_planche == planche_id)[0]
        coords_for_planche = a_scan_coords[indices]

        # Parcours des couches de la planche
        for layer_name, data in layers_data[planche_id].items():
            tree = data["tree"]
            times_list = data["times"]

            cr, cg, cb = colors_map[layer_name]

            # Requête vectorisée sur le KDTree pour tous les A-scans de cette planche
            _, nn_indices = tree.query(coords_for_planche)
            point_times_array = times_list[nn_indices]

            # Accumulation spatiale pixel par pixel
            for i_local, a_scan_idx in enumerate(indices):
                point_times = point_times_array[i_local]
                for t_ns in point_times:
                    y_pixel = int(round(t_ns / time_step_ns))
                    y_start = max(0, y_pixel - half_thick)
                    y_end = min(bscan_h, y_pixel - half_thick + thickness)

                    for dy in range(y_start, y_end):
                        final_x = a_scan_idx + margin
                        final_y = dy + margin

                        if (final_x, final_y) not in color_accumulator:
                            color_accumulator[(final_x, final_y)] = [0, 0, 0, 0]

                        acc = color_accumulator[(final_x, final_y)]
                        acc[0] += cr
                        acc[1] += cg
                        acc[2] += cb
                        acc[3] += 1

    # Appliquer les couleurs finales sur l'image
    for (x, y), (r_sum, g_sum, b_sum, count) in color_accumulator.items():
        pr, pg, pb = pixels[x, y]
        avg_cr = r_sum // count
        avg_cg = g_sum // count
        avg_cb = b_sum // count

        new_r = int(pr * (1 - alpha) + avg_cr * alpha)
        new_g = int(pg * (1 - alpha) + avg_cg * alpha)
        new_b = int(pb * (1 - alpha) + avg_cb * alpha)

        pixels[x, y] = (new_r, new_g, new_b)

    # Sauvegarde en conservant les métadonnées
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_info = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        png_info.add_text(k, v)

    rgb_img.save(out_path, "PNG", pnginfo=png_info)


def create_legend(output_dir: Path, colors_map: dict, legend_name: str):
    """Génère et sauvegarde une image de légende pour toutes les couches."""
    square_size = 128
    padding = 16
    width = 512
    font_size = 48
    height = (square_size + padding) * len(colors_map) + padding

    legend_img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(legend_img)
    font = ImageFont.load_default(size=font_size)

    for idx, (layer_name, color) in enumerate(colors_map.items()):
        y0 = padding + idx * (square_size + padding)
        x0 = padding
        y1 = y0 + square_size
        x1 = x0 + square_size

        draw.rectangle([x0, y0, x1, y1], fill=color)
        draw.text(
            (x1 + padding, y0 + (square_size - font_size) // 2),
            layer_name,
            fill="black",
            font=font
        )

    legend_img.save(output_dir / f"{legend_name}.png", "PNG")


def main():
    conf = get_conf(verbose=False)

    # Récupération des chemins de configuration sous forme de Path
    directories = [Path(p) for p in conf.input_dir]
    output_dir = Path(conf.output_dir)
    root_dir = Path(conf.root_dir)

    footprints_dict = {k: Path(v) for k, v in conf.footprint_path.items()}
    velocities_dict = {k: Path(v) for k, v in conf.velocities_path.items()}
    layers_dict = {k: [Path(p) for p in v] for k, v in conf.layers_path.items()}

    # Chargement global des données de vérité terrain par planche
    print("Chargement des configurations et fichiers de vérité terrain...")
    footprints = load_footprints(footprints_dict)
    velocities = load_velocities(velocities_dict)
    layers_data, all_layer_names = load_layers(layers_dict, velocities)

    # Attribution de couleurs uniques pour l'ensemble des couches
    raw_colors = generate_colors(len(all_layer_names))
    colors_map = {name: raw_colors[i] for i, name in enumerate(all_layer_names)}

    # Traitement itératif des images PNG
    print("Lancement du traitement des images...")
    for directory in directories:
        for img_path in directory.rglob("*.png"):
            rel_path = img_path.relative_to(root_dir)
            out_path = output_dir / rel_path

            process_image(
                img_path, out_path,
                footprints, layers_data, colors_map,
                conf.margin, conf.thickness, conf.alpha
            )

    # Création de la légende unifiée
    create_legend(output_dir, colors_map, conf.legend_name)
    print("Traitement terminé avec succès.")


if __name__ == "__main__":
    main()
