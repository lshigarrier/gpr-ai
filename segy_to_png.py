# Extracteur de tranches géoradar (GPR) de SEGY vers PNG.
#
# Ce script parcourt tous les fichiers .sgy (SEGY) d'un dossier racine
# et exporte des images PNG carrées en niveaux de gris avec des axes gradués.
#
# Logique d'extraction :
# - Fichiers 3D (C-scans) : les dimensions sont identifiées (nx antennes, nt temps, ny pas).
# - La dimension temporelle (nt) dicte la taille de la fenêtre (nt x nt).
# - Les fenêtres avancent avec un pas de nt/2 le long des traces (axe y).
# - Une dernière fenêtre est ajoutée à la fin si la longueur totale n'est
#   pas un multiple parfait du pas.
#
# Des axes sont ajoutés sur les 4 côtés :
# - Axe vertical : graduations tous les xxx ns (sans texte).
# - Axe horizontal : graduations tous les xxx cm (sans texte).
#
# Les coordonnées de la première trace et de la dernière trace de chaque
# image sont injectées dans les métadonnées (EXIF/Texte) du fichier PNG.
#
# Usage:
#     python segy_to_png.py C:\Chemin\Vers\Data

import argparse
import io
import numpy as np
import segyio
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
from pathlib import Path
from PIL import Image, PngImagePlugin


def get_real_coordinate(val, scalar):
    """Applique le scalaire de l'en-tête SEG-Y à la coordonnée brute."""
    if scalar == 0:
        return val
    elif scalar > 0:
        return val * scalar
    else:
        return val / abs(scalar)


def load_segy_cube_and_coords(sgy_path):
    """
    Charge un fichier SEGY 3D.
    Retourne le cube ordonné (nx, nt, ny), les tableaux de coordonnées X et Y (nx, ny),
    les infos de temps, le scalaire, et l'unité.
    """
    with segyio.open(str(sgy_path), "r", ignore_geometry=False) as f:
        cube = segyio.tools.cube(f)

        samples = f.samples
        nt_val = len(samples)

        # Identification des axes
        dims = list(cube.shape)
        t_axis = dims.index(nt_val)

        other_axes = [i for i in range(3) if i != t_axis]
        # x (antennes) est la plus petite dimension restante, y (acquisition) est la plus grande
        if dims[other_axes[0]] <= dims[other_axes[1]]:
            x_axis, y_axis = other_axes[0], other_axes[1]
        else:
            x_axis, y_axis = other_axes[1], other_axes[0]

        # Réorganisation en (nx, nt, ny)
        cube = np.transpose(cube, (x_axis, t_axis, y_axis))
        nx, nt, ny = cube.shape

        # Extraction de TOUTES les coordonnées pour les calquer sur la géométrie
        h0 = f.header[0]
        scalar = h0.get(segyio.TraceField.SourceGroupScalar, 1)
        # Tente de récupérer l'unité (Code 1 = Longueur (m/ft), 2 = Secondes d'arc, 3 = Degrés, etc.)
        units = h0.get(segyio.TraceField.CoordinateUnits, "Non défini")

        # On lit toutes les coordonnées X et Y dans l'ordre du fichier
        n_traces_total = f.tracecount
        all_x = np.zeros(n_traces_total)
        all_y = np.zeros(n_traces_total)
        for i in range(n_traces_total):
            h = f.header[i]
            all_x[i] = get_real_coordinate(h.get(segyio.TraceField.SourceX, 0), scalar)
            all_y[i] = get_real_coordinate(h.get(segyio.TraceField.SourceY, 0), scalar)

        # On reshape les coordonnées de la même manière que le cube (en ignorant l'axe t)
        # La géométrie d'origine de tools.cube renvoie (iline, xline, samples).
        # On la refaçonne en (nx, ny) selon l'identification des axes faite ci-dessus.
        coords_shape = list(dims)
        coords_shape.pop(t_axis)  # On retire la dimension temporelle
        all_x_2d = all_x.reshape(coords_shape)
        all_y_2d = all_y.reshape(coords_shape)

        # Transposition pour avoir (nx, ny)
        if dims.index(nx) > dims.index(ny):
            all_x_2d = all_x_2d.T
            all_y_2d = all_y_2d.T

        t_start_ns, t_end_ns = samples[0], samples[-1]
        dt_ns = samples[1] - samples[0] if len(samples) > 1 else 0

    return cube, all_x_2d, all_y_2d, t_start_ns, t_end_ns, dt_ns, scalar, units


def process_sgy_file(sgy_path, output_parent, margin_px, time_tick, dist_tick):
    """Charge un fichier SEGY, découpe le profil 3D en fenêtres carrées et génère les PNG."""
    sgy_path = Path(sgy_path)
    out_folder = output_parent / sgy_path.stem
    out_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nTraitement de : {sgy_path.name}")
    try:
        cube, coords_x, coords_y, t_start_ns, t_end_ns, dt_ns, scalar, units = load_segy_cube_and_coords(sgy_path)
    except Exception as e:
        print(f"  ERREUR lors de la lecture de {sgy_path.name} : {e}")
        return

    nx, nt, ny = cube.shape
    print(f"  -> Cube 3D identifié : nx={nx} (antennes), nt={nt} (temps), ny={ny} (pas)")

    # Calcul de la distance totale et du pas d'acquisition dx_cm
    # On se base sur l'antenne 0 (x=0) de la première à la dernière trace de l'axe y
    x0, y0 = coords_x[0, 0], coords_y[0, 0]
    x_end, y_end = coords_x[0, -1], coords_y[0, -1]

    # Unité 1 en SEG-Y correspond généralement à des mètres.
    # La distance brute est calculée en mètres, convertie ici en centimètres.
    dist_total_cm = np.sqrt((x_end - x0) ** 2 + (y_end - y0) ** 2) * 100
    if dist_total_cm == 0 or ny <= 1:
        print(f"Impossible de déterminer un pas d'acquisition géographique valide pour {sgy_path.name}. Ignoré.")
        return

    dx_cm = dist_total_cm / (ny - 1)
    window = nt
    step = nt // 2
    window_length_cm = window * dx_cm
    t_tot_ns = t_end_ns - t_start_ns

    print(f"  -> Coordonnées Trace 0 : X={x0}, Y={y0} (Scalaire: {scalar}, Unité: {units})")
    print(f"  -> Temps d'écoute total : {t_tot_ns:.2f} ns")
    print(f"  -> Pas d'écoute : {dt_ns:.2f} ns")
    print(f"  -> Longueur d'une image : {window_length_cm:.2f} cm")
    print(f"  -> Pas d'acquisition : {dx_cm:.2f} cm")

    # Détermination des index de départ pour les fenêtres carrées
    y_starts = list(range(0, max(ny - window + 1, 1), step))
    if y_starts and y_starts[-1] + window < ny:
        y_starts.append(ny - window)
    if not y_starts:
        y_starts = [0]

    saved_count = 0
    vmax = np.percentile(np.abs(cube), 99) if np.any(cube) else 1.0

    # Paramétrage strict de la résolution Matplotlib
    dpi = 100
    fig_size_px = nt + 2 * margin_px
    fig_size_inches = fig_size_px / dpi

    # Position de la zone de données (centrée)
    ax_rect = [margin_px / fig_size_px, margin_px / fig_size_px, nt / fig_size_px, nt / fig_size_px]

    # Création du gabarit unique avec Matplotlib
    fig = plt.figure(figsize=(fig_size_inches, fig_size_inches), dpi=dpi)
    ax = fig.add_axes(ax_rect)

    # On crée une image vide (blanche) pour le fond
    extent_template = [0, window_length_cm, t_end_ns, t_start_ns]
    ax.imshow(np.zeros((nt, window)), cmap="gray", vmin=0, vmax=1, extent=extent_template, aspect="auto",
              interpolation='none')

    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    ax.yaxis.set_major_locator(MultipleLocator(time_tick))

    # Les graduations X partent de 0 puisqu'elles sont toujours relatives au bord gauche
    ticks_x = np.arange(0, window_length_cm + 0.1, dist_tick)
    ax.set_xticks(ticks_x)

    # Masquer le texte des valeurs
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    # Masquer les axes
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Sauvegarde du gabarit dans un objet Image PIL en mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    template_img = Image.open(buf).convert("L")  # "L" pour Niveaux de gris

    for x_idx in range(nx):
        for y_start in y_starts:
            y_end_idx = min(y_start + window, ny)
            slice_2d = cube[x_idx, :, y_start:y_end_idx]

            # Conversion ultra-rapide du B-Scan (Numpy) en image (PIL)
            # Normalisation des valeurs entre 0 et 255 en fonction de vmax
            slice_norm = np.clip(slice_2d, -vmax, vmax)
            slice_norm = ((slice_norm + vmax) / (2 * vmax) * 255).astype(np.uint8)
            data_img = Image.fromarray(slice_norm, mode='L')

            # On copie le gabarit et on colle les données au centre
            final_img = template_img.copy()
            # On colle les pixels exactement à la marge définie
            final_img.paste(data_img, (margin_px, margin_px))

            # Coordonnées réelles pour les métadonnées (Antenne courante ou antenne 0 par défaut)
            win_x_start = coords_x[x_idx, y_start]
            win_y_start = coords_y[x_idx, y_start]
            win_x_end = coords_x[x_idx, y_end_idx - 1]
            win_y_end = coords_y[x_idx, y_end_idx - 1]

            # Ajout des métadonnées et écriture sur disque
            meta = PngImagePlugin.PngInfo()
            meta.add_text("GPS_X_START", str(win_x_start))
            meta.add_text("GPS_Y_START", str(win_y_start))
            meta.add_text("GPS_X_END", str(win_x_end))
            meta.add_text("GPS_Y_END", str(win_y_end))

            out_name = out_folder / f"{x_idx}_{y_start}.png"
            final_img.save(out_name, "PNG", pnginfo=meta)
            saved_count += 1

    print(f"  -> {saved_count} images extraites (Zone données: {nt}x{nt} px, Globale: {fig_size_px}x{fig_size_px} px).")


def find_and_process_folders(root_path, margin_px, time_tick, dist_tick):
    """Parcourt root_path avec pathlib et traite tous les fichiers .sgy trouvés."""
    root_path = Path(root_path)
    if not root_path.is_dir():
        raise ValueError(f"Le chemin spécifié n'est pas un dossier : {root_path}")

    # Recherche récursive de tous les fichiers sgy (insensible à la casse)
    sgy_files = list(root_path.rglob("*.[sS][gG][yY]"))

    if not sgy_files:
        print("Aucun fichier .sgy trouvé dans les sous-dossiers.")
        return

    print(f"Trouvé {len(sgy_files)} fichier(s) SEGY. Début du traitement...")
    for sgy_path in sgy_files:
        process_sgy_file(sgy_path, sgy_path.parent, margin_px, time_tick, dist_tick)

    print("\nTerminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrait des B-scans d'un fichier SEGY 3D vers des images PNG graduées.")
    parser.add_argument("dossier_racine", type=str,
                        help="Le chemin vers le dossier contenant les fichiers .sgy")
    parser.add_argument("-m", "--margin", type=int, default=5,
                        help="Marge blanche autour de l'image en pixels (défaut: 5)")
    parser.add_argument("-t", "--time", type=float, default=5.0,
                        help="Intervalle des graduations verticales en ns (défaut: 5)")
    parser.add_argument("-d", "--distance", type=float, default=100.0,
                        help="Intervalle des graduations horizontales en cm (défaut: 100)")

    args = parser.parse_args()

    find_and_process_folders(args.dossier_racine, args.margin, args.time, args.distance)
