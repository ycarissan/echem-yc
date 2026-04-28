"""
Utility functions for magnetic shielding calculations with PySCF-GIAO.
Contains helper functions for molecule building, shielding analysis, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyscf import gto


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Conversion factor
ANG2BOHR = 1.8897259886

# Couleurs CPK classiques pour les atomes
CPK_COLORS = {
    'C': '#404040', 'N': '#3050F8', 'O': '#FF0D0D',
    'H': '#CCCCCC', 'S': '#FFFF30', 'F': '#90E050',
    'Cl': '#1FF01F', 'default': '#FF69B4'
}

# Van der Waals radii (in Angströms)
ATOM_RADIUS = {
    'C': 0.15, 'N': 0.14, 'O': 0.13, 'H': 0.08, 'default': 0.15
}


# ═══════════════════════════════════════════════════════════════════════════════
# MOLECULE BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_molecule_with_ghosts(atom_str, points_ang, basis, ghost_basis):
    """
    Construit un objet gto.Mole contenant la molécule réelle
    + des atomes ghost (Ghost-H) aux positions demandées.
    
    Les atomes ghost ont une base mais pas d'électrons :
    ils ne modifient pas la densité électronique mais permettent
    à PySCF de calculer le tenseur de blindage à leur position.
    
    Parameters
    ----------
    atom_str : str
        Définition des atomes réels (format : "symbol x y z\n...")
    points_ang : array-like, shape (N, 3)
        Coordonnées des points fantômes en Angströms
    basis : str
        Basis set pour les atomes réels (ex: '6-311+G(d,p)')
    ghost_basis : str
        Basis set pour les atomes ghost (ex: 'sto-3g')
    
    Returns
    -------
    mol : pyscf.gto.Mole
        Molécule avec atomes réels et fantômes
    """
    ghost_lines = '\n'.join([
        f'Ghost-H  {p[0]:.8f}  {p[1]:.8f}  {p[2]:.8f}'
        for p in points_ang
    ])
    
    mol = gto.Mole()
    mol.atom = atom_str + '\n' + ghost_lines
    mol.basis = basis
    
    mol.basis = {}
    
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


# ═══════════════════════════════════════════════════════════════════════════════
# SHIELDING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_shielding(shielding_tensor):
    """
    Calcule les grandeurs dérivées du tenseur de blindage 3×3.
    
    Parameters
    ----------
    shielding_tensor : array, shape (3, 3)
        Tenseur de blindage magnétique (en ppm)
    
    Returns
    -------
    dict
        Dictionnaire contenant :
        - 'sigma_iso'   : isotropic shielding (trace/3)
        - 'sigma_xx'    : tensor component xx
        - 'sigma_yy'    : tensor component yy
        - 'sigma_zz'    : tensor component zz (perpendicular component)
        - 'NICS'        : nucleus-independent chemical shift (-sigma_iso)
        - 'NICS_zz'     : zz-component of NICS
        - 'anisotropy'  : shielding anisotropy
        - 'tensor'      : original tensor
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def get_molecule_coords(atom_str):
    """
    Parse les coordonnées atomiques depuis la string de définition.
    
    Parameters
    ----------
    atom_str : str
        Définition des atomes (format : "symbol x y z\n...")
    
    Returns
    -------
    list of dict
        Liste de dictionnaires avec clés {'symbol', 'x', 'y', 'z'}
    """
    atoms = []
    for line in atom_str.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 4:
            sym = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append({'symbol': sym, 'x': x, 'y': y, 'z': z})
    return atoms


def get_bond_pairs(atoms, max_dist_ang=1.7):
    """
    Détecte les liaisons par distance (seuil en Å).
    
    Parameters
    ----------
    atoms : list of dict
        Atomes avec coordonnées (format get_molecule_coords)
    max_dist_ang : float, optional
        Seuil de distance pour les liaisons (Angströms), default=1.7
    
    Returns
    -------
    list of tuple
        Liste de paires (i, j) d'indices d'atomes liés
    """
    bonds = []
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            a, b = atoms[i], atoms[j]
            d = np.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)
            if d < max_dist_ang:
                bonds.append((i, j))
    return bonds


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def nics_to_hex(nics_val, vmax=None):
    """
    Convertit une valeur NICS en code couleur hexadécimal.
    
    Parameters
    ----------
    nics_val : float
        Valeur NICS (en ppm)
    vmax : float, optional
        Valeur maximum pour la normalisation. Si None, utilise abs(nics_val)
    
    Returns
    -------
    str
        Code hexadécimal (#RRGGBB)
    """
    if vmax is None:
        vmax = max(abs(nics_val), 1.0)
    
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.colormaps['RdBu_r']
    rgba = cmap(norm(nics_val))
    return mcolors.to_hex(rgba)


def visualize_molecule_with_points(atoms_mol, bonds, results, nics_values):
    """
    Affiche la molécule en 3D avec py3Dmol + sphères colorées par NICS
    aux points fantômes.
    
    Parameters
    ----------
    atoms_mol : list of dict
        Atomes et leurs coordonnées (format get_molecule_coords)
    bonds : list of tuple
        Paires d'indices d'atomes liés (format get_bond_pairs)
    results : list of dict
        Résultats d'analyse pour chaque point
    nics_values : array
        Valeurs NICS pour normaliser la colormap
    """
    import py3Dmol
    
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
