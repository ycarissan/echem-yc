# ## 0. Installation des dépendances

import warnings
warnings.filterwarnings('ignore')
# Installer les packages nécessaires si besoin
import subprocess, sys

packages = ['pyscf', 'py3Dmol', 'plotly', 'numpy', 'pandas', 'matplotlib', 'jinja2', 'pyscf.prop']
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        print(f'Installation de {pkg}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

print('Toutes les dépendances sont disponibles.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px

from pyscf import gto, scf, dft
from pyscf.prop import nmr

# Constante de conversion Å → Bohr
ANG2BOHR = 1.8897259886

print('PySCF version:', gto.__version__ if hasattr(gto, '__version__') else 'OK')
print('Imports OK')

PYRAZINE_ATOM = """
C    0.000000   1.394707   0.000000
C    0.000000  -1.394707   0.000000
N    1.147376   0.718445   0.000000
N   -1.147376   0.718445   0.000000
C    1.147376  -0.718445   0.000000
C   -1.147376  -0.718445   0.000000
H    0.000000   2.479737   0.000000
H    0.000000  -2.479737   0.000000
H    2.033173   1.242049   0.000000
H   -2.033173   1.242049   0.000000
H    2.033173  -1.242049   0.000000
H   -2.033173  -1.242049   0.000000
"""

# Couleurs CPK classiques pour les atomes
CPK_COLORS = {
    'C': '#404040', 'N': '#3050F8', 'O': '#FF0D0D',
    'H': '#CCCCCC', 'S': '#FFFF30', 'F': '#90E050',
    'Cl': '#1FF01F', 'default': '#FF69B4'
}
ATOM_RADIUS = {'C': 0.15, 'N': 0.14, 'O': 0.13, 'H': 0.08, 'default': 0.15}

BASIS = '6-311+G(d,p)'   # base pour les atomes réels
GHOST_BASIS = 'sto-3g'   # base minimale pour les atomes ghost
METHOD = 'RHF'           # 'RHF' ou 'DFT-B3LYP'

# ## 3. Définition des points d'intérêt
# 
# On peut les définir directement ou les lire depuis un fichier `.xyz`.

# In[4]:


#   # ─── Option A : définition manuelle ───────────────────────────────────────────
#   points_ang = np.array([
#       [0.5,  0.5,  0.5],   # point demandé
#       [0.0,  0.0,  0.0],   # centre de la molécule (NICS(0))
#       [0.0,  0.0,  1.0],   # 1 Å au-dessus du centre
#       [0.0,  0.0, -1.0],   # 1 Å en-dessous
#       [0.0,  0.0,  2.0],   # 2 Å au-dessus
#       [1.0,  0.0,  0.0],   # dans le plan, décentré
#       [0.0,  1.0,  0.0],
#   ], dtype=float)
#   
#   # ─── Option B : lecture depuis un fichier ─────────────────────────────────────
#   # Décommenter pour lire depuis un fichier (3 colonnes : x y z, en Angström)
#   points_ang = np.loadtxt('points.xyz', comments='#')
#   
#   print(f'{len(points_ang)} points chargés :')
#   for i, p in enumerate(points_ang):
#       print(f'  [{i}]  x={p[0]:7.3f}  y={p[1]:7.3f}  z={p[2]:7.3f}  Å')


def build_molecule_with_ghosts(atom_str, points_ang, basis, ghost_basis):
    """
    Construit un objet gto.Mole contenant la molécule réelle
    + des atomes ghost (Ghost-H) aux positions demandées.

    Les atomes ghost ont une base mais pas d'électrons :
    ils ne modifient pas la densité électronique mais permettent
    à PySCF de calculer le tenseur de blindage à leur position.
    """
    ghost_lines = '\n'.join([
        f'Ghost-H  {p[0]:.8f}  {p[1]:.8f}  {p[2]:.8f}'
        for p in points_ang
    ])

    mol = gto.Mole()
    mol.atom = atom_str + '\n' + ghost_lines
    mol.basis = basis  # PySCF applique la même base à tous

    # Spécifier la base minimale pour les ghosts
    # On utilise un dict pour différencier
    mol.basis = {}

    # Parser l'atom_str pour récupérer les symboles uniques
    real_symbols = set()
    for line in atom_str.strip().split('\n'):
        if line.strip():
            sym = line.split()[0]
            real_symbols.add(sym)

    for sym in real_symbols:
        mol.basis[sym] = basis
    mol.basis['Ghost-H'] = ghost_basis

    mol.unit = 'Angstrom'
    mol.symmetry = False
    mol.verbose = 3
    mol.build()

    return mol


#   mol = build_molecule_with_ghosts(PYRAZINE_ATOM, points_ang, BASIS, GHOST_BASIS)
#   
#   n_real = len([l for l in PYRAZINE_ATOM.strip().split('\n') if l.strip()])
#   n_ghost = len(points_ang)
#   print(f'\nMolécule construite : {n_real} atomes réels + {n_ghost} atomes ghost')
#   print(f'Nombre total de fonctions de base : {mol.nao}')
#   
#   
#   # ## 5. Calcul SCF
#   
#   # In[6]:
#   
#   
#   if METHOD == 'RHF':
#       mf = scf.RHF(mol)
#   elif METHOD == 'DFT-B3LYP':
#       mf = dft.RKS(mol)
#       mf.xc = 'b3lyp'
#   else:
#       raise ValueError(f'Méthode inconnue : {METHOD}')
#   
#   mf.max_cycle = 200
#   mf.conv_tol = 1e-10
#   
#   print(f'Lancement du calcul {METHOD}...')
#   energy = mf.kernel()
#   print(f'\nÉnergie SCF : {energy:.10f} Hartree')
#   print(f'Convergé : {mf.converged}')
#   
#   
#   # ## 6. Calcul NMR — tenseur de blindage magnétique
#   
#   # In[7]:
#   
#   
#   print('Calcul du tenseur de blindage magnétique (GIAO)...')
#   
#   if METHOD == 'RHF':
#       from pyscf.prop.nmr import rhf as nmr_mod
#       mf_nmr = nmr_mod.NMR(mf)
#   elif METHOD == 'DFT-B3LYP':
#       from pyscf.prop.nmr import rks as nmr_mod
#       mf_nmr = nmr_mod.NMR(mf)
#   
#   # Calcul de tous les tenseurs (atomes réels + ghosts)
#   shielding_all = mf_nmr.kernel()
#   
#   print(f'\nCalcul terminé. Tenseurs obtenus pour {len(shielding_all)} centres.')
#   
#   
#   # ## 7. Extraction et analyse des résultats
#   
#   # In[8]:


def analyze_shielding(shielding_tensor):
    """Calcule les grandeurs dérivées du tenseur de blindage 3×3."""
    # Partie symétrique
    sigma_sym = 0.5 * (shielding_tensor + shielding_tensor.T)

    # Valeurs propres (composantes principales)
    eigenvalues = np.linalg.eigvalsh(sigma_sym)
    sigma_11, sigma_22, sigma_33 = np.sort(eigenvalues)  # σ11 ≤ σ22 ≤ σ33

    sigma_iso = np.trace(shielding_tensor) / 3.0
    sigma_zz  = shielding_tensor[2, 2]   # composante perpendiculaire au plan moléculaire
    aniso     = sigma_33 - 0.5 * (sigma_11 + sigma_22)  # anisotropie

    return {
        'sigma_iso': sigma_iso,
        'sigma_xx':  shielding_tensor[0, 0],
        'sigma_yy':  shielding_tensor[1, 1],
        'sigma_zz':  sigma_zz,
        'NICS':     -sigma_iso,
        'NICS_zz':  -sigma_zz,
        'anisotropy': aniso,
        'tensor':   shielding_tensor,
    }

#   # ─── Résultats sur les atomes ghost ───────────────────────────────────────────
#   results = []
#   print('=' * 75)
#   print(f'{"Point":>4}  {"x":>7} {"y":>7} {"z":>7}  {"σ_iso":>10}  {"NICS":>10}  {"NICS_zz":>10}')
#   print('=' * 75)
#   
#   for i, point in enumerate(points_ang):
#       idx = n_real + i  # indice dans le tableau de blindage
#       sigma = shielding_all[idx]
#       res = analyze_shielding(sigma)
#       res['x'], res['y'], res['z'] = point
#       res['point_idx'] = i
#       results.append(res)
#   
#       print(f'[{i:2d}]  {point[0]:7.3f} {point[1]:7.3f} {point[2]:7.3f}  '
#             f'{res["sigma_iso"]:10.3f}  {res["NICS"]:10.3f}  {res["NICS_zz"]:10.3f}')
#   
#   print('=' * 75)
#   print('Toutes les valeurs en ppm.')
#   
#   # ─── Résultats sur les noyaux réels ───────────────────────────────────────────
#   print('\n--- Blindage sur les noyaux réels ---')
#   atom_symbols = [mol.atom_symbol(i) for i in range(n_real)]
#   for i in range(n_real):
#       sigma = shielding_all[i]
#       sigma_iso = np.trace(sigma) / 3.0
#       print(f'  Atome {i:2d} ({atom_symbols[i]:2s}): σ_iso = {sigma_iso:8.3f} ppm')
#   
#   
#   # ## 8. Tableau récapitulatif
#   
#   # In[9]:
#   
#   
#   df = pd.DataFrame([
#       {
#           'Point': f"[{r['point_idx']}]",
#           'x (Å)': r['x'],
#           'y (Å)': r['y'],
#           'z (Å)': r['z'],
#           'σ_iso (ppm)': round(r['sigma_iso'], 3),
#           'σ_xx (ppm)':  round(r['sigma_xx'], 3),
#           'σ_yy (ppm)':  round(r['sigma_yy'], 3),
#           'σ_zz (ppm)':  round(r['sigma_zz'], 3),
#           'NICS (ppm)':    round(r['NICS'], 3),
#           'NICS_zz (ppm)': round(r['NICS_zz'], 3),
#       }
#       for r in results
#   ])
#   
#   # Mise en forme avec gradient de couleur sur NICS
#   df.style.background_gradient(
#       subset=['NICS (ppm)', 'NICS_zz (ppm)'],
#       cmap='RdBu_r',
#       vmin=-20, vmax=20
#   ).format(precision=3)
#   
#   
#   # In[10]:
#   
#   
#   get_ipython().system('pip install nbformat')
#   
#   
#   # ## 9. Visualisation 3D interactive
#   # 
#   # La molécule est affichée en fil de fer (stick model) et les points de calcul sont représentés par des sphères colorées selon la valeur du NICS.
#   # 
#   # **Convention de couleur :**
#   # - 🔵 Bleu → NICS négatif → **blindage, zone aromatique (courants de cycle diamagnétiques)**
#   # - ⚪ Blanc → NICS ≈ 0 → zone neutre
#   # - 🔴 Rouge → NICS positif → **déblindage, zone antiaromatique ou en dehors**
#   
#   # In[11]:


def get_molecule_coords(atom_str):
    """Parse les coordonnées atomiques depuis la string de définition."""
    atoms = []
    for line in atom_str.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 4:
            sym = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append({'symbol': sym, 'x': x, 'y': y, 'z': z})
    return atoms


def get_bond_pairs(atoms, max_dist_ang=1.7):
    """Détecte les liaisons par distance (seuil en Å)."""
    bonds = []
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            a, b = atoms[i], atoms[j]
            d = np.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)
            if d < max_dist_ang:
                bonds.append((i, j))
    return bonds


def nics_to_hex(nics_val):
    rgba = cmap(norm(nics_val))
    return mcolors.to_hex(rgba)


def visualize_molecule_with_points(atoms_mol, bonds, results, nics_values):
    """
    Affiche la molécule en 3D avec py3Dmol + sphères colorées par NICS
    aux points fantômes.
    """
    # ── Colormap NICS ──────────────────────────────────────────────────────────
    vmax = max(abs(nics_values).max(), 1.0)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.colormaps['RdBu_r']

    def nics_to_rgb(val):
        r, g, b, _ = cmap(norm(val))
        return f'0x{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

    # ── Construction du viewer ─────────────────────────────────────────────────
    view = py3Dmol.view(width=800, height=600)

    # -- Molécule : générer un bloc XYZ minimal
    xyz_str = f"{len(atoms_mol)}\npyrazine\n"
    for a in atoms_mol:
        xyz_str += f"{a['symbol']}  {a['x']:.6f}  {a['y']:.6f}  {a['z']:.6f}\n"

    view.addModel(xyz_str, 'xyz')
    view.setStyle({'model': 0}, {
        'stick': {'radius': 0.12, 'color': 'gray'},
        'sphere': {'scale': 0.25}   # atomes en sphères réduites
    })

    # -- Sphères aux points fantômes, colorées par NICS
    for r in results:
        color = nics_to_rgb(r['NICS'])
        view.addSphere({
            'center': {'x': r['x'], 'y': r['y'], 'z': r['z']},
            'radius': 0.3,
            'color': color,
            'opacity': 0.85
        })
        # Label avec la valeur NICS
        view.addLabel(
            f"NICS={r['NICS']:.1f}",
            {
                'position': {'x': r['x'], 'y': r['y'], 'z': r['z']},
                'fontSize': 10,
                'fontColor': 'black',
                'backgroundOpacity': 0.4,
                'backgroundColor': 'white'
            }
        )

    view.zoomTo()
    view.show()

    # ── Colorbar matplotlib à part ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 0.5))
    fig.subplots_adjust(bottom=0.5)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, orientation='horizontal'
    )
    cb.set_label('NICS (ppm)  —  Bleu = blindé | Rouge = déblindé', fontsize=10)
    plt.show()
