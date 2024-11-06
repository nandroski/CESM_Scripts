#!/usr/bin/env python3
import xarray as xr
import numpy as np
import metpy as mp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Literal
import cftime

class Midpoint_Normalize(colors.Normalize):
    '''
    Normalizes a colorbar so that diverging bars that work 
    around prescribed midpoint value.

    Code from Joe Kington (github: joferkington)
    Found this at https://chris35wills.github.io/matplotlib_diverging_colorbar/
    Relatively simple but gets the job done.
    '''
    def __init__(self, vmin=None,vmax=None, midpoint=None,clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self,vmin,vmax,clip)
    def __call__(self,value,clip=None):
        x,y = [self.vmin,self.midpoint,self.vmax],[0,0.5,1]
        return np.ma.masked_array(np.interp(value,x,y),np.isnan(value))


# 2D contour plotter
# make a function that inputs a netCDF file, a parameter of interest, horz/vertical axes (from a choice of lat,lon,lev), axes of mean (the choice not chosen)
# inputs for title, levels, extend flag, filename, aspect ratio, cbar_tick array, center flag, save flag, cmap flag, label fontsize, plot size in inches

# add a unit option in the future
def plot_contour(input_file,parameter,horz_ax: Literal['lon', 'lat', 'lev'],vert_ax: Literal['lon', 'lat', 'lev'],
                cmap='cubehelix',title=None,levels=10,output_file=None,show=True,save=False,flip_y=False,
                plot_size=(12,8),axlabel_size=14,titlelabel_size=16,aspect_ratio="2:1",cbar_ticks=None,
                extend=False,center=False,log_y=False,nyear=None,
                derived_function=None,derived_input_list=None):
    '''
    Plots a 2D contourf plot for a given parameter from a single netCDF file along two axes, taking a mean across the other.
    Ex: if horz_ax = 'lon' and vert_ax = 'lat', then the parameter will be averaged along lev. 
    If a time axis is given, the parameter will be averaged along it as well.

    Input:
    -----------------
    input_file : filepath to netCDf file
    parameter : string of variable name in CDF file
    horz_ax : variable to be used as horizontal axis (either 'lon','lat', or 'lev')
    vert_ax : variable to be used as vertical axis (either 'lon','lat', or 'lev')
    cmap (default = 'cubehelix') : colormap used for contour plot
    title (default = None) : title for plot
    levels (default = 10): levels used for contourf. If integer, uses that many contours, 
                           if array plots contours for the values. If None given then 10
                           equidistant contours are calculated using max and min.
    output_file (default = None) : output filename used when saving plot
    show (default = True) : if True, plot.show() is run
    save (default = False) : if True, saves plot as png using output_file as filename
    flip_y (default = False) : if True, flips y axis
    plot_size (default = (12,8)) : tuple containing the width and height in inches.
    axlabel_size (default = 14) : fontsize of axes labels and ticks
    titlelabel_size (default = 16) : fontsize of title
    aspect_ratio (default = '2:1') : string containing the aspect ratio width:height to be enforced for plot area
    cbar_ticks (default = None) : An array containing the tick number labels to be used for the colorbar.
    extend (default = False) : If True, adds triangles above and below colorbar to account for over/undersaturated values
    center (default = False) : If True, centers the colorbar around 0 so that diverging colormaps will properly 
                               align even with asymetric max and min levels.
    log_y (default = False) : if True, makes y-axis log10 scale
    nyear (default = None) : The integer refering to the year to start time average 
    derived_function (default = None) : function handle for a derived quantity that can be calculated with variables in file.
                                       If provided derived_input_list is required and parameter variable is ignored.
    derived_input_list (default = None) : list of strings of variables names required for derived_function given in the
                                         same order as the derived_function handle definition.

    Ex of derived_function:
    Say we have a potential temperature function pot_temp that has inputs pressure and temperature.
    def pot_temp(p,T):
        return T(1000/p)**(R_d/c_p)
    So we input derived_function = pot_temp and derived_input_list = ['lev','T']
    Preferably the functions should list variables like pressure, lat, lon first.
    
    

    '''


    
    
    # determine axis that mean will be calculated along
    mean_axis = ['lon','lat','lev']
    mean_axis.remove(horz_ax)
    mean_axis.remove(vert_ax)
    mean_axis = mean_axis[0]
    
    # set max and min levels
    if type(levels) is not int and levels is not None:
        max_level = levels[-1]
        min_level = levels[0]
    else:
        max_level=None
        min_level=None


    if nyear is None:
        start_time = cftime.DatetimeNoLeap(1,1,1,has_year_zero=True)
    else:
        start_time = cftime.DatetimeNoLeap(nyear,1,1,has_year_zero=True)
    
    # extract aspect ratio
    width = int(aspect_ratio.split(':')[0])
    height = int(aspect_ratio.split(':')[1])

    # if not plot size is given, calculate based on aspect ratio
    if plot_size is None:
        plot_size = (4*width,4*height)
    
    #read in file and extract parameters
    dataset = xr.open_dataset(input_file)
    dataset = dataset.metpy.parse_cf()

    if derived_function is not None and derived_input_list is not None:
        derived_input = []
        for i in derived_input_list:
            derived_input.append(dataset[i])
        Z = derived_function(*derived_input)
        X,Y = np.meshgrid(derived_input[-1].coords[horz_ax].values,derived_input[-1].coords[vert_ax].values)
        if 'time' in derived_input[-1].coords:
            Z = Z[Z.time>=start_time].mean('time')
    else:
        Z = dataset[parameter]
        if 'time' in Z.coords:
            Z = Z[Z.time>=start_time].mean('time')
        X,Y = np.meshgrid(Z.coords[horz_ax].values,Z.coords[vert_ax].values)

    plot_parameter = Z.mean(mean_axis)
    print(f'plotting {mean_axis}-mean {parameter} from {input_file}')
    max_value = np.max(plot_parameter)
    min_value = np.min(plot_parameter)
    mean_value = np.mean(plot_parameter)
    order_of_mag = np.floor(np.log10(mean_value)).values
    print(order_of_mag)
    print(f'Max Value: {max_value:.2E}')
    print(f'Min Value: {mean_value:.2E}')
    print(f'Mean Value: {min_value:.2E}')

    if levels is None:
        # let's have 10 levels by default
        level_num = 10
        if order_of_mag < 0:
            max_level = np.ceil(max_value.values*10**(-order_of_mag))/(10**order_of_mag)
            min_level = np.floor(min_value.values*10**(-order_of_mag))/(10**order_of_mag)
        else:
            max_level = np.ceil(max_value.values)
            min_level = np.floor(min_value.values)
        step =  (max_level-min_level)/10
        levels = np.arange(min_level,
                           max_level+step,
                           step)


    
    #define figure
    fig,ax = plt.subplots(1,1,figsize=plot_size)

    #plot contour
    if center and extend:
        CF = ax.contourf(X,Y,plot_parameter,levels = levels,cmap=cmap,
                         norm=Midpoint_Normalize(midpoint=0,vmin=min_level,vmax=max_level),
                         extend='both')
    elif center:
        CF = ax.contourf(X,Y,plot_parameter,levels = levels,cmap=cmap,
                         norm=Midpoint_Normalize(midpoint=0,vmin=min_level,vmax=max_level))
    elif extend:
        CF = ax.contourf(X,Y,plot_parameter,levels = levels,cmap=cmap,extend='both')        
    else:
        CF = ax.contourf(X,Y,plot_parameter,levels = levels,cmap=cmap)

    if flip_y:
        plt.gca().invert_yaxis()

    #format to ensure aspect ratio is maintained
    ax.set_box_aspect(height/width)
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.3)  
    # Colorbar with 5% width of the plot
    
    if center:
        norm = colors.BoundaryNorm(levels,  plt.get_cmap(cmap).N, clip=True)
        cbar = fig.colorbar(CF,cax=cbar_ax,boundaries=levels, ticks=levels,
                            spacing='proportional')
    else:
        cbar = fig.colorbar(CF,cax=cbar_ax)


    cbar.ax.tick_params(labelsize=axlabel_size)
    
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)

    
    if horz_ax == 'lat':
        ax.set_xticks(np.linspace(-90,90,7))
        ax.set_xlabel("Latitude",fontsize=axlabel_size)
    elif horz_ax == 'lon':
        ax.set_xticks(np.linspace(0,360,7))
        ax.set_xlabel("Longitude",fontsize=axlabel_size)
    else:
        ax.set_xticks(np.linspace(np.floor(X.min()),np.ceil(X.max()),10))
        ax.set_xlabel("Pressure [hPa]",fontsize=axlabel_size)

    if vert_ax == 'lat':
        ax.set_yticks(np.linspace(-90,90,7))
        ax.set_ylabel("Latitude",fontsize=axlabel_size)
    elif vert_ax == 'lon':
        ax.set_yticks(np.linspace(0,360,7))
        ax.set_ylabel("Longitude",fontsize=axlabel_size)
    elif log_y:
        plt.yscale("log")
        ax.set_ylabel("Pressure [hPa]",fontsize=axlabel_size)
    else:
        ax.set_yticks(np.linspace(np.floor(Y.min()),np.ceil(Y.max()),10))
        ax.set_ylabel("Pressure [hPa]",fontsize=axlabel_size)


    ax.tick_params(axis='both', which='major', labelsize=axlabel_size)
    ax.set_title(title,fontsize=titlelabel_size)
    fig.tight_layout()
    
    if save and output_file is not None:
        fig.savefig(f"{output_file}.png",dpi=300)
    if show:
        plt.show()
    return

