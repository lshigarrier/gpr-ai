"""
Objectif :
Ce script génère des points de mesure le long de lignes parallèles au sein d'une emprise définie par 4 coins.
Il crée 14 lignes partant du segment [coin_2, coin_3] en direction du segment [coin_1, coin_4].
Sur chaque ligne, des points sont générés tous les 10 cm. Les coordonnées (converties en mètres via EPSG:32631
pour le calcul) sont ensuite reconverties en degrés (EPSG:4326). Enfin, les épaisseurs associées à chaque ligne
(lues dans un fichier technique) sont ajoutées et le résultat complet est sauvegardé dans un fichier CSV.

Entrées :
- emprise_csv (Path) : Chemin vers le CSV contenant les coordonnées des coins (id, latitude, longitude).
- manuel_csv (Path) : Chemin vers le CSV contenant les épaisseurs (id, layer_1, layer_2, layer_3, layer_4).

Sorties :
- output_csv (Path) : Chemin vers le CSV généré contenant les colonnes latitude, longitude, layer_1, layer_2, layer_3, layer_4.

Exemple de commande :
python generate_measurements.py emprise_souple.csv manuel_technique_epaisseurs.csv --output resultats.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer


def load_and_project_corners(emprise_path: Path) -> dict:
    """Charge les coins et convertit leurs coordonnées de l'EPSG:4326 vers EPSG:32631."""
    df_corners = pd.read_csv(emprise_path, index_col='id')

    # Transformer EPSG:4326 (lat, lon) -> EPSG:32631 (x, y en mètres)
    transformer_to_m = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)

    # Attention: always_xy=True signifie que l'entrée doit être (longitude, latitude)
    lons = df_corners['longitude'].values
    lats = df_corners['latitude'].values
    xs, ys = transformer_to_m.transform(lons, lats)

    corners = {
        idx: np.array([x, y])
        for idx, x, y in zip(df_corners.index, xs, ys)
    }
    return corners


def compute_intersection_distances(starts: np.ndarray, u_dir: np.ndarray, c1: np.ndarray, c4: np.ndarray) -> np.ndarray:
    """
    Calcule la distance maximale (t) pour chaque point de départ avant de croiser le segment [c1, c4].
    Utilise la règle de Cramer pour résoudre vectoriellement les intersections de lignes.
    """
    # Vecteur du segment cible
    v41 = c4 - c1

    # Matrice du système A = [u_dir, -v41]
    # det(A) = ux * (-v41y) - uy * (-v41x)
    det_A = u_dir[0] * (-v41[1]) - u_dir[1] * (-v41[0])

    # Vecteurs b = c1 - start pour chaque point de départ
    b_x = c1[0] - starts[:, 0]
    b_y = c1[1] - starts[:, 1]

    # t = det(A_t) / det(A)
    # A_t = [b, -v41] => det(A_t) = b_x * (-v41y) - b_y * (-v41x)
    det_A_t = b_x * (-v41[1]) - b_y * (-v41[0])

    t_distances = det_A_t / det_A
    return t_distances


def generate_lines_points(corners: dict, step: float = 0.1) -> pd.DataFrame:
    """Génère tous les points au pas spécifié sur les 14 lignes."""
    c1, c2, c3, c4 = corners['coin_1'], corners['coin_2'], corners['coin_3'], corners['coin_4']

    # 1. Détermination des points de départ (15 points de c2 à c3, on exclut le premier)
    w_vec = c3 - c2
    fractions = np.linspace(0, 1, 15)[1:]  # Exclut 0 (coin_2), garde les 14 autres jusqu'à 1 (coin_3)
    start_points = c2 + fractions[:, np.newaxis] * w_vec

    # 2. Vecteur directionnel (de coin_2 vers coin_1)
    dir_vec = c1 - c2
    u_dir = dir_vec / np.linalg.norm(dir_vec)

    # 3. Calcul des distances max pour chaque ligne jusqu'au segment [coin_1, coin_4]
    max_distances = compute_intersection_distances(start_points, u_dir, c1, c4)

    # 4. Génération des points (vectorisation par concaténation de plages)
    all_points = []
    line_ids = []

    for i, (start, max_dist) in enumerate(zip(start_points, max_distances)):
        # Génère des distances de 0 jusqu'à max_dist (exclus) avec un pas de 0.1m
        distances = np.arange(0, max_dist, step)

        # Coordonnées des points pour cette ligne : Start + d * u_dir
        points = start + distances[:, np.newaxis] * u_dir
        all_points.append(points)

        # Les ID commencent à 2 et vont de 2 en 2
        line_id = (i + 1) * 2
        line_ids.extend([line_id] * len(distances))

    all_points_array = np.vstack(all_points)

    df_points = pd.DataFrame({
        'id': line_ids,
        'x': all_points_array[:, 0],
        'y': all_points_array[:, 1]
    })

    return df_points


def main():
    parser = argparse.ArgumentParser(description="Génère des points de mesure avec épaisseurs associés.")
    parser.add_argument('footprint_path', type=Path, help="Chemin vers emprise_souple.csv")
    parser.add_argument('thickness_path', type=Path, help="Chemin vers manuel_technique_epaisseurs.csv")
    parser.add_argument("--output", type=Path, default=Path("layers.csv"), help="Chemin du CSV de sortie.")
    args = parser.parse_args()

    # 1. Chargement et projection des coins
    corners = load_and_project_corners(args.footprint_path)

    # 2. Génération des coordonnées en mètres (EPSG:32631)
    df_points = generate_lines_points(corners, step=0.1)
    EPSG: 32631
    # 3. Reprojection des points vers EPSG:4326 (lat, lon)
    transformer_to_deg = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    lons, lats = transformer_to_deg.transform(df_points['x'].values, df_points['y'].values)
    df_points['longitude'] = lons
    df_points['latitude'] = lats

    # 4. Jointure avec le manuel technique
    df_manuel = pd.read_csv(args.thickness_path)

    # Fusion des données sur la colonne 'id'
    df_final = df_points.merge(df_manuel, on='id', how='left')

    # 5. Sélection des colonnes finales et export
    columns_to_keep = ['latitude', 'longitude', 'layer_1', 'layer_2', 'layer_3', 'layer_4']
    df_final = df_final[columns_to_keep]

    df_final.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
