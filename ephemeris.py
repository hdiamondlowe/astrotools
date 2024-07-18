# calculate transit (or eclipse) times - ingress, egress, midpoint
# Author: Hannah Diamond-Lowe
# Date: 26 Feb. 2016
# Updated: 2 Feb. 2016  Now you can make pretty airmass graphs that show where the transit/eclipse is

# want to add RA and Dec to this in order to calculate transit & eclipse times for non-circular orbits

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import astropy.units as u
from astropy.io import ascii
import pytz
import datetime
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon

def event_time_calculator(input_file, start_time='now', end_time='week', output=False, bad_transit_flag=True, make_latex_table=False, obstimes=False):


    params = ascii.read(input_file)
    if start_time == 'now': start_time=Time.now().jd
    elif type(start_time) == str: start_time = Time(start_time).jd
    else: 
        print('start time must be string of format 2017-12-31T00:00:00')
        return

    if end_time == 'week': end_time = start_time + 7
    elif type(end_time) == str: end_time = Time(end_time).jd
    else:
        print('end time must be string of format 2017-12-31T00:00:00')
        return


    if end_time < start_time:
        print('end time must come after start time')
        #return

    for p in params:
        name = p['name']
        ra = p['ra_hms']
        dec = p['dec_dms']
        event = p['event']
        #f.write(name + '(' + event + ')')
        t0 = Time(p['midpoint_BJDtdb'], format='jd', scale='utc').value
        period = p['period_days']
        locname = p['location']
        tdur = p['duration_days']
        
        start_num = int(round((start_time - t0)/period))
        end_num = int(round((end_time - t0)/period))
        event_num = np.arange(start_num, end_num+1)
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='fk5')
        utc = pytz.timezone('UTC')   # sets up the utc timezone
        if locname == 'LCO': lat, lon, height, tz = -29.0146*u.deg, -70.6926*u.deg, 2380.*u.m, pytz.timezone('America/Santiago')
        if locname == 'VLT': lat, lon, height, tz = -24.62733*u.deg, -70.40417*u.deg, 2635*u.m, pytz.timezone('America/Santiago')
        if locname == 'MMT': lat, lon, height, tz = 31.67*u.deg, -110.88*u.deg, 2617.*u.m, pytz.timezone('America/Phoenix')
        if locname == 'GeminiN': lat, lon, height, tz = 19.82396*u.deg, -155.46984*u.deg, 4213.*u.m, pytz.timezone('US/Hawaii')
        if locname != 'Space': loc = EarthLocation.from_geodetic(lat=lat, lon=lon, height=height)


        if 'transit' in event:
            mdpts = [t0 + n*period for n in event_num]
        elif 'eclipse' in event:
            mdpts = [t0 + n*period + 0.5*period for n in event_num] # assumes e=0; so far not set up for e != 0
        else:
            print('Must identify type of event (Tra or Ecl)')
            continue

        event_mdpts = Time(mdpts, format='jd', scale='utc')

        if locname == 'Space':

            for n, mdpt in enumerate(event_mdpts):
                print(mdpt.iso)
                deltamdpt = np.linspace(-2.5*tdur, 2.5*tdur, 500)*u.day
                utctime = (mdpt.value*u.day)+deltamdpt

                transitinds = np.where((utctime > (mdpt.value*u.day)-(tdur/2.*u.day)) & (utctime < (mdpt.value*u.day)+(tdur/2.*u.day)))

                # if any of the transit is below airmass 2, or it is transiting during sun-up don't bother plotting
                #if bad_transit_flag: 
                #    if np.any(airmass[transitinds] > 2) or np.any(airmass[transitinds] < 0): 
                #        print('skip: airmass, avg secz = {0}'.format(np.mean(airmass[transitinds])))
                #        continue
                #    if np.any(sun.alt[transitinds].value > 0): 
                #        print('skip: sun is up')
                #        continue
                #    good_event_mdpts_inds.append(n)


                if obstimes:
                    observationtimes.setdefault('mdpt', []).append(mdpt)
                    observationtimes.setdefault('T1', []).append(mdpt-((0.5*tdur)*u.day))
                    observationtimes.setdefault('T4', []).append(mdpt+((0.5*tdur)*u.day))
                    observationtimes.setdefault('obsstart', []).append(mdpt-((1*tdur)*u.day)-(11*u.min))
                    observationtimes.setdefault('obsend', []).append(mdpt+((1*tdur)*u.day))

                fig = plt.figure(figsize=(8, 8))
                ax1 = fig.add_subplot(111)

                # instead of airmass just like a line??
                ax1.axhline(1, 0, 1, color='mediumblue', lw=2)
                ax1.plot(utctime[transitinds], [1 for i in utctime[transitinds]], 'k', lw=10, alpha=0.5)
                ax1.set_xlim(utctime.value[0], utctime.value[-1])
                #plt.ylim(1, 3)

                label_format = '{:%H:%M:%S}'
                ticks_loc = ax1.get_xticks().tolist()
                ax1.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                labels = Time(ticks_loc, format='jd', scale='utc')
                ax1.set_xticklabels([label_format.format(x) for x in labels.datetime])
                ax1.set_xlim(utctime.value[0], utctime.value[-1])

                ax1.grid(alpha=0.4)
                plt.title('{} {} mid-point {:%Y-%m-%d %H:%M:%S}'.format(locname, event, mdpt.datetime), y=1.08, fontsize=16)
                plt.tight_layout()


                if output:

                    f = open(name + '_events.txt', 'w')
                    f.write('name(event)    num     event_midpoint_utc     event_midpoint_jd      telescope\n')
                    [f.write(name + '(' + event + ')      '
                            + str(event_num[i]) + '      '
                            + str(event_mdpts[i].iso) +  '      '
                            + str(event_mdpts[i].jd) +  '     ' 
                            + locname + '     ' 
                            '\n') for i in range(len(event_mdpts))]

                    plt.savefig('{}_{}_{}_{:%y%m%d_%H:%M:%S}.png'.format(name, event, locname, mdpt.datetime))
                    plt.close()


                else: 
                    print(p['name'])
                    for i in range(len(event_num)):
                        print(event_num[i], event_mdpts.iso[i])

                    plt.show()
                    plt.close()

            continue


        if bad_transit_flag: good_event_mdpts_inds = []
        if obstimes: observationtimes = {}
        min_airmass = []
        solar_altitudes = []
        for n, mdpt in enumerate(event_mdpts):
            print(mdpt.iso)
            deltamdpt = np.linspace(-2.5*tdur, 2.5*tdur, 500)*u.day
            objframe = AltAz(obstime=mdpt+deltamdpt, location=loc)
            objevent = coord.transform_to(objframe)
            airmass = objevent.secz

            utctime = (mdpt.value*u.day)+deltamdpt
            transitinds = np.where((utctime > (mdpt.value*u.day)-(tdur/2.*u.day)) & (utctime < (mdpt.value*u.day)+(tdur/2.*u.day)))# & (airmass > 1) & (airmass < 3))
            airmassinds = np.where((airmass > 1) & (airmass < 3))

            sun = get_sun(Time(mdpt+deltamdpt, format='jd', scale='utc')).transform_to(objframe)
            min_airmass.append(np.min(airmass[transitinds]))

            objframe_mdpt = AltAz(obstime=mdpt, location=loc)
            sun_mdpt = get_sun(Time(mdpt, format='jd', scale='utc')).transform_to(objframe_mdpt)
            solar_altitudes.append(sun_mdpt.alt)

            moon = get_moon(Time(mdpt+deltamdpt, format='jd', scale='utc')).transform_to(objframe)
            moon_mdpt = get_moon(Time(mdpt, format='jd', scale='utc')).transform_to(objframe_mdpt)
            mdpt_separation = moon_mdpt.separation(coord).deg


            # if any of the transit is below airmass 2, or it is transiting during sun-up don't bother plotting
            if bad_transit_flag: 
                if np.any(airmass[transitinds] > 2) or np.any(airmass[transitinds] < 0): 
                    print('skip: airmass, avg secz = {0}'.format(np.mean(airmass[transitinds])))
                    continue
                if np.any(sun.alt[transitinds].value > 0): 
                    print('skip: sun is up')
                    continue
                good_event_mdpts_inds.append(n)


            if obstimes:
                observationtimes.setdefault('mdpt', []).append(mdpt)
                observationtimes.setdefault('T1', []).append(mdpt-((0.5*tdur)*u.day))
                observationtimes.setdefault('T4', []).append(mdpt+((0.5*tdur)*u.day))
                observationtimes.setdefault('obsstart', []).append(mdpt-((1*tdur)*u.day)-(11*u.min))
                observationtimes.setdefault('obsend', []).append(mdpt+((1*tdur)*u.day))

            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()

            ax1.plot(utctime[airmassinds], airmass[airmassinds], color='mediumblue', lw=2)
            ax1.plot(utctime[transitinds], airmass[transitinds], 'k', lw=10, alpha=0.5)
            ax1.set_xlim(utctime.value[0], utctime.value[-1])
            plt.ylim(1, 3)

            ax1.plot(utctime[::50], moon.secz[::50], 'ko', markersize=12, markeredgewidth=0, alpha=0.3, label='sep. at mid-transit:\n{0:.2f} deg'.format(mdpt_separation))
            ax1.legend(loc='lower right', fontsize=12)

            label_format = '{:%H:%M:%S}'
            ticks_loc = ax1.get_xticks().tolist()
            ax1.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            labels = Time(ticks_loc, format='jd', scale='utc')
            ax1.set_xticklabels([label_format.format(x) for x in labels.datetime])
            ax1.set_xlim(utctime.value[0], utctime.value[-1])
            labels_local = [utc.localize(label).astimezone(tz) for label in labels.datetime]
            ax2.set_xticklabels([label_format.format(x) for x in labels_local])
            ax2.set_xlim(utctime.value[0], utctime.value[-1])

            # plot gradients of twilight
            ax1.fill_between(utctime.value, 1, 3, sun.alt > -0*u.deg, color='goldenrod', alpha=0.5, zorder=0)
            ax1.fill_between(utctime.value, 1, 3, sun.alt > -6*u.deg, color='goldenrod', alpha=0.5, zorder=0)
            ax1.fill_between(utctime.value, 1, 3, sun.alt > -12*u.deg, color='goldenrod', alpha=0.5, zorder=0)
            ax1.fill_between(utctime.value, 1, 3, sun.alt > -18*u.deg, color='goldenrod', alpha=0.5, zorder=0)

            plt.gca().invert_yaxis()

            # figure out the dates for the beginning and end of the night
            localtime = utc.localize(mdpt.datetime).astimezone(tz)
            ax1.set_xlabel('Time (UTC)', fontsize=14)
            ax2.set_xlabel('Local Time (UTC{0}:{1})'.format(localtime.strftime('%z')[:-2], localtime.strftime('%z')[-2:]), fontsize=14)
            ax1.set_ylabel('Airmass (sec(z))', fontsize=14)

            local_night_start, local_night_end = night_start_end(mdpt, tz)

            ax1.grid(alpha=0.4)
            plt.title('{} {:%Y-%m-%d} ---- {:%Y-%m-%d}'.format(locname, local_night_start, local_night_end), y=1.08, fontsize=16)
            plt.tight_layout()
            if output == True:
                plt.savefig('{}_{}_{}_{:%y%m%d}-{:%d}_{:%H:%M:%S}.png'.format(name, event, locname, local_night_start, local_night_end, mdpt.datetime))
                plt.close()
            else:
                plt.show()
                plt.close()

        if output and not bad_transit_flag:
            f = open('{}_{}_events.txt'.format(names, locname), 'w')
            f.write('name(event)    num     event_midpoint_utc      event_ingress     event_egress      air_mass      solar_altitude      telescope\n')
            [f.write(name + '(' + event + ')      '
                    + str(event_num[i]) + '      '
                    + str(event_mdpts[i].iso) + '     ' 
                    + str(min_airmass[i]) +  '     ' 
                    + locname +  '     ' 
                    '\n') for i in range(len(event_mdpts))]

        elif output and bad_transit_flag:
            f = open('{}_{}_events.txt'.format(name, locname), 'w')
            for i in good_event_mdpts_inds:
                local_midtransit_time = utc.localize(event_mdpts[i].datetime).astimezone(tz)
                local_night_start, local_night_end = night_start_end(event_mdpts[i], tz)

                if i == 0: f.write('name(event)    num     evend_midpoint_jd       event_midpoint_utc      local_night       event_midpoint_local     air_mass      solar_altitude      telescope\n')
                f.write(name + '(' + event + ')      '
                        + str(event_num[i]) + '      '
                        + str(event_mdpts[i].jd) + '     ' 
                        + str(event_mdpts[i].iso) + '     ' 
                        + '{:%y:%m:%d}'.format(local_night_start) + '---' 
                        + '{:%y:%m:%d}'.format(local_night_end) + '     ' 
                        + '{:%H:%M:%S}'.format(local_midtransit_time) + '     ' 
                        + str(round(min_airmass[i].value, 2)) +  '     ' 
                        + locname +  '     ' 
                        '\n')
            if make_latex_table:
                f = open('{}_events_{}_latex.tex'.format(name, locname), 'w')

                f.write('\\documentclass[12pt]{article} \n')
                f.write('\\begin{document} \n')

                f.write('\\begin{table}[h!] \n')
                f.write('\\begin{center} \n')
                f.write('\\begin{tabular}{c|c|c|c} \n')
                f.write('Mid-transit       & Local Night   & Mid-transit    & Peak  Airmass        \\\\ \n')
                f.write('(UTC)             & (LCO time)    & (LCO time)     & (sec\\textit{z})           \\\\ \n')
                f.write('    \\hline \n')

                for i in good_event_mdpts_inds:
                    local_night_start, local_night_end = night_start_end(event_mdpts[i], tz)
                    f.write('     {:%y%m%d_:%d:%H:%M:%S}'.format(event_mdpts[i].datetime) +  '    &'  + '{:%y:%m:%d}'.format(local_night_start) + '---' + '{:%y:%m:%d}'.format(local_night_end) + '    &' + '{:%H:%M:%S}'.format(local_midtransit_time.datetime) + '    &' + str(round(min_airmass[i].value, 2)) + '\\\\ \n')
                 
                f.write('\\end{tabular} \n')
                f.write('\\end{center} \n')
                f.write('\\end{table} \n')
                f.write('\\end{document} \n')

                f.close()

        else:
            print(p['name'])
            for i in range(len(event_num)):
                print(event_num[i], event_mdpts.iso[i])

        if obstimes:

            f = open('{}_obstimes_{}.txt'.format(name, locname), 'w')

            f.write('Observation Night    Start Observations       T1         Mid-Transit    T4         End Observations\n')

            for i in range(len(observationtimes['mdpt'])):

                mdpt = observationtimes['mdpt'][i]
                gmt_tz = pytz.timezone('GMT')

                # not actually local; now in utc
                mdpt_local = utc.localize(mdpt.datetime).astimezone(gmt_tz)
                local_night_start, local_night_end = night_start_end(mdpt, gmt_tz)
                obsstart_local = utc.localize(observationtimes['obsstart'][i].datetime).astimezone(gmt_tz)
                T1_local = utc.localize(observationtimes['T1'][i].datetime).astimezone(gmt_tz)
                T4_local = utc.localize(observationtimes['T4'][i].datetime).astimezone(gmt_tz)
                obsend_local = utc.localize(observationtimes['obsend'][i].datetime).astimezone(gmt_tz)

                f.write('ut{:%y%m%d}_{:%d}          {:%H:%M:%S}                 {:%H:%M:%S}   {:%H:%M:%S}       {:%H:%M:%S}   {:%H:%M:%S}\n'.format(local_night_start, local_night_end, obsstart_local, T1_local, mdpt_local, T4_local, obsend_local))

            f.close()

def night_start_end(event_mdpt, timezone):
    utc = pytz.timezone('UTC')   # sets up the utc timezone
    local_midtransit_time = utc.localize(event_mdpt.datetime).astimezone(timezone)
    if local_midtransit_time.hour >= 12: 
        local_night_start = local_midtransit_time
        local_night_end = local_midtransit_time + datetime.timedelta(days=1)
    if local_midtransit_time.hour < 12:
        local_night_start = local_midtransit_time - datetime.timedelta(days=1)
        local_night_end = local_midtransit_time

    return local_night_start, local_night_end

