import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from pyproj import Transformer


def load_and_project_emprise(csv_path: Path) -> dict:
    """
    Charge le fichier CSV d'emprises et convertit les coordonnées EPSG:4326 (lat/lon) vers EPSG:32631 (mètres).
    """
    df_emprise = pd.read_csv(csv_path)

    # always_xy=True garantit que l'ordre est (longitude, latitude) -> (X, Y)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)

    corners = {}
    for _, row in df_emprise.iterrows():
        x, y = transformer.transform(row['Longitude'], row['Latitude'])
        corners[row['ID']] = np.array([x, y])

    return corners


def compute_geometry(corners: dict):
    """
    Calcule les 12 points de départ et le vecteur directeur unitaire.
    """
    c1 = corners['Coin_1']
    c2 = corners['Coin_2']
    c4 = corners['Coin_4']

    # Création de 12 points intérieurs sur un segment divisé en 13 intervalles
    start_points = [c1 + (i / 13.0) * (c4 - c1) for i in range(1, 13)]

    # Vecteur directeur unitaire (Coin_1 -> Coin_2)
    vec_c1_c2 = c2 - c1
    direction_vector = vec_c1_c2 / np.linalg.norm(vec_c1_c2)

    return start_points, direction_vector


def process_excel(excel_path: Path, start_points: list, direction_vector: np.ndarray) -> pd.DataFrame:
    """
    Parcourt l'Excel, associe les onglets R et B, et calcule les coordonnées 3D.
    """
    xls = pd.ExcelFile(excel_path, engine='openpyxl')
    results = []

    # On isole les onglets qui finissent par 'R' et font 4 caractères
    r_sheets = [s for s in xls.sheet_names if len(s) == 4 and s.endswith('R')]

    col_names = ['Distance', 'B', 'C', 'D', 'E', 'F', 'G']

    for r_sheet in r_sheets:
        b_sheet = r_sheet[:-1] + 'B'

        if b_sheet not in xls.sheet_names:
            raise ValueError(f"L'onglet '{b_sheet}' correspondant à '{r_sheet}' est introuvable.")

        # Détermination de l'offset pour les points de départ
        second_char = r_sheet[1]
        if second_char == '1':
            offset = 0  # Départs 1 à 6
        elif second_char == '2':
            offset = 6  # Départs 7 à 12
        else:
            raise ValueError(f"Le second caractère de '{r_sheet}' doit être 1 ou 2. Trouvé: '{second_char}'")

        # Lecture des DataFrames
        df_r = pd.read_excel(xls, sheet_name=r_sheet, header=None, skiprows=2, usecols="A:G", names=col_names)
        df_b = pd.read_excel(xls, sheet_name=b_sheet, header=None, skiprows=2, usecols="A:G", names=col_names)

        # On supprime les lignes où la distance est vide et on met la distance en index
        df_r = df_r.dropna(subset=['Distance']).set_index('Distance')
        df_b = df_b.dropna(subset=['Distance']).set_index('Distance')

        # Jointure sur la Distance (index). On garde la base R (left join)
        df_merged = df_r.join(df_b, lsuffix='_R', rsuffix='_B')

        # Parcours des distances
        for distance, row in df_merged.iterrows():

            # Parcours des colonnes B à G
            for col_idx, col_letter in enumerate(['B', 'C', 'D', 'E', 'F', 'G']):
                epaisseur_r = row[f"{col_letter}_R"]

                # Récupération sécurisée de l'épaisseur B (au cas où l'index n'existerait pas dans B)
                col_b_name = f"{col_letter}_B"
                epaisseur_b = row[col_b_name] if col_b_name in row else np.nan

                # Si R ou B est vide, on ignore
                if pd.isna(epaisseur_r) or pd.isna(epaisseur_b):
                    continue

                # Calcul de la position X, Y
                start_point = start_points[offset + col_idx]
                current_pos = start_point + distance * direction_vector

                # Création du point
                results.append({
                    "coord_x": current_pos[0],
                    "coord_y": current_pos[1],
                    "layer_1": epaisseur_r,
                    "layer_2": epaisseur_b
                })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Extrait et géoréférence les épaisseurs Excel vers profondeurs CSV.")
    parser.add_argument("excel_path", type=Path, help="Chemin vers le fichier Excel.")
    parser.add_argument("emprises_path", type=Path, help="Chemin vers le fichier emprises.csv.")
    parser.add_argument("--output", type=Path, default=Path("layers.csv"),
                        help="Chemin du CSV de sortie.")
    args = parser.parse_args()

    corners = load_and_project_emprise(args.emprises_path)
    start_points, direction_vector = compute_geometry(corners)

    df_results = process_excel(args.excel_path, start_points, direction_vector)

    df_results.to_csv(args.output, index=False)
    print(f"Succès : {len(df_results)} points sauvegardés dans {args.output}")


if __name__ == "__main__":
    main()
