"""
Objectif du code :
Ce script génère un fichier CSV contenant des coordonnées géographiques (EPSG:4326) de points
d'inspection de voirie et leurs épaisseurs de chaussée associées (layer_1, layer_2).
Il lit un fichier d'emprise pour définir des segments de départ, détermine l'intersection
de vecteurs directeurs avec ces segments pour trouver le point de départ exact de chaque ligne,
puis calcule la position spatiale des mesures d'épaisseur lues dans un fichier Excel.

Entrées :
- footprint_path : CSV contenant les coins de l'emprise (id, latitude, longitude)
- vector_path : CSV contenant les coordonnées des lignes (id, lat1, lon1, lat2, lon2)
- excel_path : Excel contenant les distances et les épaisseurs par ligne
- --output : (Optionnel) Chemin du fichier CSV de sortie

Sortie :
- Un fichier CSV contenant les colonnes : latitude, longitude, layer_1, layer_2

Exemple de commande :
python generate_layers.py emprise_souple.csv lignes_roadscanners.csv "épaisseurs de chaussée.xlsx" --output resultats_epaisseurs.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer


def get_line_intersection(p1, v_dir, seg_A, seg_B):
    """
    Calcule l'intersection entre une ligne définie par un point (p1) et un vecteur directeur (v_dir)
    et un segment défini par deux points (seg_A, seg_B).
    """
    v_seg = seg_B - seg_A

    # Résolution du système linéaire: p1 + t * v_dir = seg_A + u * v_seg
    # t * v_dir - u * v_seg = seg_A - p1
    A = np.column_stack((v_dir, -v_seg))
    b = seg_A - p1

    # On résout pour [t, u]
    t_u = np.linalg.solve(A, b)
    t = t_u[0]

    # Le point d'intersection
    intersection = p1 + t * v_dir
    return intersection


def process_sheet_vectorized(df_sheet, start_point, v_dir_norm):
    """
    Traite un dataframe pandas correspondant à un onglet de l'Excel, filtre et arrange
    les colonnes D et E, et calcule les coordonnées projetées le long de la ligne.
    """
    # Renommer pour simplifier la manipulation (colonnes: 0 -> distance, 3 -> D, 4 -> E)
    df_sheet.columns = ['distance', 'D', 'E']

    # Filtrer les lignes où les deux épaisseurs sont 0
    mask_both_zero = (df_sheet['D'] == 0) & (df_sheet['E'] == 0)
    df_valid = df_sheet.loc[~mask_both_zero].copy()

    # Application de la logique d'assignation vectorisée pour layer_1 et layer_2
    df_valid['layer_1'] = np.where(df_valid['D'] == 0, df_valid['E'], df_valid['D'])
    df_valid['layer_2'] = np.where(df_valid['D'] == 0, 0, df_valid['E'])

    # Calcul des coordonnées X, Y en mètres (vectorisé)
    distances = df_valid['distance'].values
    x_coords = start_point[0] + distances * v_dir_norm[0]
    y_coords = start_point[1] + distances * v_dir_norm[1]

    df_valid['X'] = x_coords
    df_valid['Y'] = y_coords

    return df_valid[['X', 'Y', 'layer_1', 'layer_2']]


def main():
    parser = argparse.ArgumentParser(description="Génère un CSV des points géolocalisés avec leurs épaisseurs.")
    parser.add_argument('footprint_path', type=Path, help="Chemin vers emprise_souple.csv")
    parser.add_argument('vector_path', type=Path, help="Chemin vers lignes_roadscanners.csv")
    parser.add_argument('excel_path', type=Path, help="Chemin vers le fichier Excel des épaisseurs")
    parser.add_argument('--output', type=Path, default=Path("layers.csv"), help="Chemin du CSV de sortie.")
    args = parser.parse_args()

    # Initialisation des transformateurs de coordonnées
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    transformer_to_wgs = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)

    # 1. Lecture et conversion de l'emprise
    df_footprint = pd.read_csv(args.footprint_path)
    # Transformation (lon, lat) -> (X, Y)
    emprise_x, emprise_y = transformer_to_utm.transform(df_footprint['longitude'].values,
                                                        df_footprint['latitude'].values)

    # Extraction des coins (indexation numpy par ID ou par position, on suppose coin_1, coin_2, coin_4 aux index logiques)
    # On isole les coordonnées des points nécessaires
    p_coin1 = np.array([emprise_x[df_footprint['id'] == 'coin_1'][0], emprise_y[df_footprint['id'] == 'coin_1'][0]])
    p_coin2 = np.array([emprise_x[df_footprint['id'] == 'coin_2'][0], emprise_y[df_footprint['id'] == 'coin_2'][0]])
    p_coin4 = np.array([emprise_x[df_footprint['id'] == 'coin_4'][0], emprise_y[df_footprint['id'] == 'coin_4'][0]])

    # 2. Lecture et conversion des vecteurs de ligne
    df_vectors = pd.read_csv(args.vector_path)

    x1, y1 = transformer_to_utm.transform(df_vectors['lon1'].values, df_vectors['lat1'].values)
    x2, y2 = transformer_to_utm.transform(df_vectors['lon2'].values, df_vectors['lat2'].values)

    df_vectors['X1'] = x1
    df_vectors['Y1'] = y1
    df_vectors['X2'] = x2
    df_vectors['Y2'] = y2

    all_results = []

    # 3. Parcours et traitement de chaque ligne
    for _, row in df_vectors.iterrows():
        line_id = int(row['id'])

        p1 = np.array([row['X1'], row['Y1']])
        p2 = np.array([row['X2'], row['Y2']])
        v_dir = p2 - p1
        v_dir_norm = v_dir / np.linalg.norm(v_dir)

        # Choix du segment cible en fonction de la ligne
        if line_id in [1, 2]:
            seg_A, seg_B = p_coin1, p_coin4
        else:  # Lignes 3, 4, 5, 6
            seg_A, seg_B = p_coin1, p_coin2

        # Calcul du point de départ sur le segment
        start_point = get_line_intersection(p1, v_dir, seg_A, seg_B)

        # 4. Lecture et traitement des données Excel pour cette ligne
        sheet_name = f"Ligne {line_id}"
        # On utilise les index de colonnes 0 (A), 3 (D) et 4 (E)
        df_sheet = pd.read_excel(args.excel_path, sheet_name=sheet_name, header=0, usecols=[0, 3, 4])

        df_processed = process_sheet_vectorized(df_sheet, start_point, v_dir_norm)
        all_results.append(df_processed)

    # 5. Concaténation et reprojection finale
    df_final = pd.concat(all_results, ignore_index=True)

    final_lon, final_lat = transformer_to_wgs.transform(df_final['X'].values, df_final['Y'].values)
    df_final['longitude'] = final_lon
    df_final['latitude'] = final_lat

    # 6. Export du fichier
    df_export = df_final[['latitude', 'longitude', 'layer_1', 'layer_2']]
    df_export.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
