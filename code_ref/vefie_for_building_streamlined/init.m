function input = init()

input.f = 60e6; % frequency (Hz)
% If object_gen is 'Yes' then the function to generate the raw geometry
% programmatically will be called. Otherwise pre-generated files will be
% used. If object_gen is 'No' then pre-generated geometry files will be used.
input.object_gen = 'No';

% Discretisation Settings
input.disc_per_lambda = 10; % chosen accuracy

% Pass list of materials to be used in scene if it is to be generated from scratch.
input.object_materials = table({'vacuum','concrete','wood','glass','brick'}','VariableNames',{'Name'});

% Pass list of master materials with associated visualisation colourings to be used in scene to the generator.
Name = {'vacuum','concrete','wood','glass','brick','plasterboard','ceiling-board','chipboard','floorboard','metal'}';
hex = ['#FDFF00'; '#A7A9AB'; '#D08931'; '#B9E8E9'; '#ED774C'; '#EFEFEE'; '#F4F1DB'; '#C09B53'; '#7A5409'; '#909090'];
Number = 1:1:length(Name);
input.materials_master = table(Number',Name,hex);

input.directory_geom = './Geometry/';
input.object_name = 'object_room.txt';
% input.object_name = 'object_mp_landscape.txt';
end
