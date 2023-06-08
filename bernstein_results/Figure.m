%   Figure.m
%   ========
%   Creates a figure of a neat size, and handles also some options
%   associated with printing the whole stuff (all arguments are optional)...
%
%   Usage: handle = Figure(n, size, fontname, fontsize, quiet, orient)
%       n         : figure number (default=1)
%       size      : dimensions of figure [nx, ny] centimeters (default=[16, 12])
%       fontname  : fontname (default='Helvetica')
%       fontsize  : fontsize (default=12)
%       quiet     : 'quiet' be quiet, 'verbose' be verbose (default='verbose')
%       orient    : 'landscape' or 'portrait' (default='portrait')
%

function handle = Figure(n, dims, fontname, fontsize, quiet, orie);

% define non-existing parameters
if ~exist('dims'), dims=[16, 12]; end
if ~exist('n'), n=1; end
if ~exist('fontname'), fontname='Helvetica'; end
if ~exist('fontsize'), fontsize=12; end
if ~exist('quiet'), quiet='verbose'; end
if ~exist('orie'), orie='portrait'; end

% set/get default parameters
ssz = get(0, 'ScreenSize');	% screen size
ppi = 72;					% points per inch
inch = 2.54;				% cm
point = inch/ppi;			% cm
if (strcmp(orie, 'portrait') == 1)
	a4x = 21/inch;			% A4-size portrait
	a4y = 29.7/inch;		% A4-size portrait
else
	a4x = 29.7/inch;		% A4-size landscape
	a4y = 21/inch;			% A4-size landscape
	orie = 'landscape';
end


% calculate parameters
dx1 = (5*point*fontsize)/dims(1);
dy1 = (4*point*fontsize)/dims(2);
dims = dims/inch;
posx = (a4x-dims(1))/2;
posy = (a4y-dims(2))/2;
maxx = ssz(3);
maxy = ssz(4);
factor = min(min(maxx/dims(1), maxy/dims(2)), 72);
dx = factor*dims(1);
dy = factor*dims(2);

% set defaults
if strcmp(quiet,'verbose'),
	fprintf('Creating figure centered on A4 paper\n');
end

if ishandle(n), delete(n); end
handle = figure(n);
set(0, 'DefaultAxesFontName', fontname);
set(0, 'DefaultAxesFontSize', fontsize);
set(0, 'DefaultAxesPosition', [dx1, dy1, 1-0.05-dx1, 1-0.1-dy1]);
set(handle,...
	'PaperUnits', 'inches',...
	'PaperOrientation', orie,...
	'PaperPosition', [posx, posy, dims(1), dims(2)],...
	'PaperPositionMode', 'manual',...
	'PaperType', 'a4',...
	'Position', [(maxx-dx)/2, (maxy-dy)/2, dx, dy]);
