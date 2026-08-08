import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib.colors import Normalize, LogNorm
import numpy.typing as npt
import typing

DEFAULT_FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'figures'))
Path(DEFAULT_FIG_DIR).mkdir(parents=True, exist_ok=True)

def visualiztion_one2one_3D(self, fields_prior: npt.NDArray[np.float32], fields_pred: npt.NDArray[np.float32],
                            sims: int, property_name: str, plot_range: typing.Tuple = (126, 125, 110),
                            figname: str = "visualiztion_one2one_3D", savefig: bool = True):
    """
    Visualize 3D fields for prior and predicted results side by side, and save them as images.
    """
    x_range, y_range, z_range = plot_range
    field_prior = self.get_field(fields_prior, property_name, sims, x_range, y_range, z_range)
    field_pred = self.get_field(fields_pred, property_name, sims, x_range, y_range, z_range)
    norm = Normalize(vmin=field_prior.min(), vmax=field_prior.max()) if property_name == "PORO" \
        else LogNorm(vmin=field_prior.min(), vmax=field_prior.max())

    prior_figname = os.path.join(DEFAULT_FIG_DIR, f"{figname}_prior_{property_name}_{sims}_{x_range}x{y_range}x{z_range}.png")
    pred_figname = os.path.join(DEFAULT_FIG_DIR, f"{figname}_pred_{property_name}_{sims}_{x_range}x{y_range}x{z_range}.png")

    self.plot_3D_surface(
        data=field_pred,
        property_name=property_name,
        norm=norm,
        figname=pred_figname,
        savefig=savefig
    )
    self.plot_3D_surface(
        data=field_prior,
        property_name=property_name,
        norm=norm,
        figname=prior_figname,
        savefig=savefig
    )

def plot_3D_surface(data: npt.NDArray[np.float32], property_name: str, norm, figname: str = "plot_3D_surface.png", savefig: bool = True):
    """
    Plot a 3D surface of the given data and save the visualization.
    Returns the matplotlib figure object for further manipulation.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    label_fontsize = 10
    title_fontsize = 12
    ax.set_zlabel('Cell Grid ID (Z)', fontsize=label_fontsize, rotation=90)
    cmap = plt.get_cmap('viridis')

    def plot_surface(array, x, y, z):
        ax.plot_surface(x, y, z, facecolors=cmap(norm(array)), rstride=1, cstride=1, shade=False)

    nx, ny, nz = data.shape
    z = 0
    y, x = np.meshgrid(np.arange(ny + 1), np.arange(nx + 1))
    plot_surface(np.pad(data[:, :, z], ((0, 1), (0, 1)), mode="edge"), x, y, np.full_like(x, z))
    y = ny - 1
    z, x = np.meshgrid(np.arange(nz + 1), np.arange(nx + 1))
    plot_surface(np.pad(data[:, y, :], ((0, 1), (0, 1)), mode="edge"), x, np.full_like(x, ny), z)
    x = nx - 1
    z, y = np.meshgrid(np.arange(nz + 1), np.arange(ny + 1))
    plot_surface(np.pad(data[x, :, :], ((0, 1), (0, 1)), mode="edge"), np.full_like(y, nx), y, z)
    z_trim, y_trim = np.meshgrid(np.arange(nz + 1), np.arange(ny // 2, ny + 1))
    plot_surface(np.pad(data[0, ny // 2:, :], ((0, 1), (0, 1)), mode="edge"), np.full_like(z_trim, 0), y_trim, z_trim)

    ax.set_xlim([0, nx])
    ax.set_ylim([0, ny])
    ax.set_zlim([0, nz])
    ax.set_xlabel('Cell Grid ID (X)', fontsize=label_fontsize)
    ax.set_ylabel('Cell Grid ID (Y)', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=label_fontsize)
    ax.set_title(f'{property_name}', fontsize=title_fontsize)
    plt.tight_layout()
    if savefig:
        save_path = os.path.join(DEFAULT_FIG_DIR, figname) if not os.path.isabs(figname) else figname
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, transparent=True)
    return fig

def plot_comparison_and_compute_errors(head, head_solved, figname: str = "plot_comparison_and_compute_errors.png", savefig: bool = True):
    """
    Plot comparison between Matlab and Python FEM solutions and compute error metrics.
    Returns the matplotlib figure object for further manipulation.
    """
    Path(DEFAULT_FIG_DIR).mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    hmin, hmax = np.min(head), np.max(head)
    lvls = np.linspace(-10, -0.1, 7)
    cmp_str = 'RdBu'
    im1 = ax1.pcolormesh(head, cmap='viridis', vmin=hmin, vmax=hmax)
    CT1 = ax1.contour(head, levels=lvls, cmap=cmp_str)
    ax1.clabel(CT1, fontsize=15, inline=True, inline_spacing=1, fmt='%.1f')
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Matlab FEM')
    im2 = ax2.pcolormesh(head_solved, cmap='viridis', vmin=hmin, vmax=hmax)
    CT2 = ax2.contour(head_solved, levels=lvls, cmap=cmp_str)
    ax2.clabel(CT2, fontsize=15, inline=True, inline_spacing=1, fmt='%.1f')
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('Python FEM')
    plt.tight_layout()
    if savefig:
        save_path = os.path.join(DEFAULT_FIG_DIR, figname) if not os.path.isabs(figname) else figname
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_flux_map_streamlines(head_solved, qx, qy, dx, dy, figname: str = "plot_flux_map_streamlines.png", savefig: bool = True):
    """
    Plot the flux map using streamlines.
    Returns the matplotlib figure object for further manipulation.
    """
    numnodx, numnody = head_solved.shape
    x = np.linspace(0, dx * (numnodx - 1), numnodx)
    y = np.linspace(0, dy * (numnody - 1), numnody)
    X, Y = np.meshgrid(x, y)
    speed = np.sqrt(qx**2 + qy**2)
    x_mid = np.linspace(0, dx * (numnodx - 2), numnodx-1)
    y_mid = np.linspace(0, dy * (numnody - 2), numnody-1)
    X_mid, Y_mid = np.meshgrid(x_mid, y_mid)
    Path(DEFAULT_FIG_DIR).mkdir(exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, head_solved, levels=20, cmap='viridis', alpha=1)
    plt.colorbar(label="Hydraulic Head")
    plt.streamplot(X_mid, Y_mid, qy, qx, color='black', density=[0.5, 2], linewidth=1, broken_streamlines=True)
    plt.title("Streamlines with Hydraulic Head Contours")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    if savefig:
        save_path = os.path.join(DEFAULT_FIG_DIR, figname) if not os.path.isabs(figname) else figname
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, transparent=True)
    return plt.gcf()

def plot_history(history, figname: str = "plot_history.png", savefig: bool = True):
    """
    Plot optimization history including loss, lambda, step norm and computation time.
    Returns the matplotlib figure object for further manipulation.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    ax1.semilogy(history['loss'])
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss History')
    ax1.grid(True)
    ax2.semilogy(history['lambda'])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Lambda')
    ax2.set_title('Lambda History')
    ax2.grid(True)
    ax3.semilogy(history['step_norm'])
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Step Norm')
    ax3.set_title('Step Norm History')
    ax3.grid(True)
    ax4.plot(history['time'])
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Computation Time')
    ax4.grid(True)
    plt.tight_layout()
    if savefig:
        save_path = os.path.join(DEFAULT_FIG_DIR, figname) if not os.path.isabs(figname) else figname
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    return fig

def plot_parameter_history(history, V, beta=0, figname: str = "plot_parameter_history.png", savefig: bool = True):
    """
    Plot the evolution of the parameter field during optimization.
    Returns the matplotlib figure object for further manipulation.
    """
    n_iters = len(history['b'])
    n_cols = min(5, n_iters)
    n_rows = (n_iters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes)
    for i in range(n_iters):
        row, col = divmod(i, n_cols)
        s = V.T @ history['b'][i][:, np.newaxis] + beta
        im = axes[row, col].imshow(s.reshape(-1, int(np.sqrt(len(s)))), cmap='jet')
        axes[row, col].set_title(f'Iteration {i}')
        plt.colorbar(im, ax=axes[row, col])
    for i in range(n_iters, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        fig.delaxes(axes[row, col])
    plt.tight_layout()
    if savefig:
        save_path = os.path.join(DEFAULT_FIG_DIR, figname) if not os.path.isabs(figname) else figname
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    return fig

def plot_conductivity_fields(reconstructed_field, true_field, nx, ny, figname: str = "plot_conductivity_fields.png", savefig: bool = True):
    """
    Plot and compare reconstructed and true conductivity fields.
    Returns the matplotlib figure object for further manipulation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    vmin = min(reconstructed_field.min(), true_field.min())
    vmax = max(reconstructed_field.max(), true_field.max())
    im1 = ax1.pcolormesh(reconstructed_field.reshape((nx, ny)), cmap='jet', vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Reconstructed Conductivity Field')
    im2 = ax2.pcolormesh(true_field.reshape((nx, ny)), cmap='jet', vmin=vmin, vmax=vmax)
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('True Conductivity Field')
    plt.tight_layout()
    if savefig:
        save_path = os.path.join(DEFAULT_FIG_DIR, figname) if not os.path.isabs(figname) else figname
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_parameters(true_alpha, predicted_alpha, figname: str = "plot_parameters.png", savefig: bool = True):
    """
    Create a 45-degree cross-plot comparing true and predicted parameters.
    Returns the matplotlib figure object for further manipulation.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    min_val = min(np.min(true_alpha), np.min(predicted_alpha))
    max_val = max(np.max(true_alpha), np.max(predicted_alpha))
    buffer = (max_val - min_val) * 0.1
    ax.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 'k--', alpha=0.5, label='Perfect Match')
    ax.scatter(true_alpha, predicted_alpha, alpha=0.6)
    ax.set_xlabel('True Parameters')
    ax.set_ylabel('Predicted Parameters')
    ax.set_title('Cross-plot of True vs Predicted Parameters')
    ax.set_aspect('equal')
    ax.set_xlim(min_val-buffer, max_val+buffer)
    ax.set_ylim(min_val-buffer, max_val+buffer)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if savefig:
        save_path = os.path.join(DEFAULT_FIG_DIR, figname) if not os.path.isabs(figname) else figname
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_observations_vs_predictions(true_heads, predicted_heads, figname: str = "plot_observations_vs_predictions.png", savefig: bool = True):
    """
    Create a 45-degree cross-plot comparing true and predicted hydraulic heads.
    Returns the matplotlib figure object for further manipulation.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    min_val = min(np.min(true_heads), np.min(predicted_heads))
    max_val = max(np.max(true_heads), np.max(predicted_heads))
    buffer = (max_val - min_val) * 0.1
    ax.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 'k--', alpha=0.5, label='Perfect Match')
    ax.scatter(true_heads, predicted_heads, alpha=0.6)
    ax.set_xlabel('True Hydraulic Heads')
    ax.set_ylabel('Predicted Hydraulic Heads')
    ax.set_title('Cross-plot of True vs Predicted Hydraulic Heads')
    ax.set_aspect('equal')
    ax.set_xlim(min_val-buffer, max_val+buffer)
    ax.set_ylim(min_val-buffer, max_val+buffer)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if savefig:
        save_path = os.path.join(DEFAULT_FIG_DIR, figname) if not os.path.isabs(figname) else figname
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_head_fields(true_heads, predicted_heads, figname: str = "plot_head_fields.png", savefig: bool = True):
    """
    Plot and compare true and predicted hydraulic head fields.
    Returns the matplotlib figure object for further manipulation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    vmin = min(np.min(true_heads), np.min(predicted_heads))
    vmax = max(np.max(true_heads), np.max(predicted_heads))
    im1 = ax1.imshow(true_heads, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title('True Hydraulic Head Field')
    im2 = ax2.imshow(predicted_heads, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title('Predicted Hydraulic Head Field')
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    plt.tight_layout()
    if savefig:
        save_path = os.path.join(DEFAULT_FIG_DIR, figname) if not os.path.isabs(figname) else figname
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_transient_head_comparison(head_true, head_solved, time_idx=0, lvls=None, cmp_str='RdBu'):
    """
    Plot comparison between true and solved head distributions.
    """
    assert head_true.shape == head_solved.shape, "head_true and head_solved must have the same shape"
    assert len(head_true.shape) == 3, "head_true and head_solved must be 3D arrays"
    if lvls is None:
        lvls = np.linspace(-10, -0.1, 7)
    hmin, hmax = np.min(head_true), np.max(head_true)

    head_true_at_time = head_true[time_idx, :, :]
    head_solved_at_time = head_solved[time_idx, :, :]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    im1 = ax1.pcolormesh(head_true_at_time, cmap='viridis', vmin=hmin, vmax=hmax)
    CT1 = ax1.contour(head_true_at_time, levels=lvls, cmap=cmp_str)
    ax1.clabel(CT1, fontsize=15, inline=True, inline_spacing=1, fmt='%.1f')
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Matlab FEM time step {}'.format(time_idx))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    im2 = ax2.pcolormesh(head_solved_at_time, cmap='viridis', vmin=hmin, vmax=hmax)
    CT2 = ax2.contour(head_solved_at_time, levels=lvls, cmap=cmp_str)
    ax2.clabel(CT2, fontsize=15, inline=True, inline_spacing=1, fmt='%.1f')
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('Python FEM time step {}'.format(time_idx))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.tight_layout()
    return fig

def plot_trainsient_hydraulic_heads(transient_heads, t_max=1.0, hmin=-10.0, hmax=0.1, nlvls=0, cmp_str='viridis'):
    """
    Plot transient hydraulic heads for multiple wells.
    """
    assert len(transient_heads.shape) == 3, "HT_transient_heads must be a 3D array"

    if nlvls > 0:
        lvls = np.linspace(hmin, hmax, nlvls)
    else:
        lvls = np.linspace(hmin, hmax, 7)

    fig, axs = plt.subplots(1, transient_heads.shape[0], figsize=(46, 5))
    dt = t_max / transient_heads.shape[0]
    for i in range(transient_heads.shape[0]):
        axs[i].set_title('Time step {}'.format(i))
        axs[i].set_xlabel('X')
        axs[i].set_ylabel('Y')
        im = axs[i].pcolormesh(transient_heads[i, :, :], cmap=cmp_str, vmin=hmin, vmax=hmax)
        axs[i].set_title('Time: {:.2f} h'.format(i*dt))
        CT = axs[i].contour(transient_heads[i, :, :], levels=lvls, cmap=cmp_str)
        axs[i].clabel(CT, fontsize=10, inline=True, fmt='%.1f')
        fig.colorbar(im, ax=axs[i])   
    plt.tight_layout()
    return fig

def plot_transient_hydraulic_heads_at_timestep(transient_heads, t_idx=1, dt=0.1, hmin=None, hmax=None, nlvls=0, cmp_str='viridis'):
    """
    Plot transient hydraulic heads for multiple wells in 2D.
    """
    assert len(transient_heads.shape) == 3, "transient_heads must be a 3D array"
    if hmin is None:
        hmin = np.min(transient_heads)
    if hmax is None:
        hmax = np.max(transient_heads)
    if nlvls > 0:
        lvls = np.linspace(hmin, hmax, nlvls)
    else:
        lvls = np.linspace(hmin, hmax, 7)

    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    im = axs.pcolormesh(transient_heads[t_idx, :, :], cmap=cmp_str, vmin=hmin, vmax=hmax)
    CT = axs.contour(transient_heads[t_idx, :, :], levels=lvls, cmap="jet")
    axs.clabel(CT, fontsize=10, inline=True, fmt='%.1f')
    fig.colorbar(im, ax=axs)
    axs.set_title('Time: {:.2f} h'.format((t_idx+1)*dt))
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    nx, ny, nz = 16, 16, 8
    K = np.exp(np.random.rand(nx, ny, nz)-4)
    norm = LogNorm(vmin=K.min(), vmax=K.max())
    plot_3D_surface(K, "PERM", norm)
