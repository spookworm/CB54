function image_object_render(image_object, materials_master, markerColor, title_str)
image_object_render = ind2rgb(image_object, materials_master.('map'));
figure
imagesc(image_object_render, 'XData', 1/2, 'YData', 1/2)
title(title_str)
legend
set(gcf, 'units', 'normalized', 'outerposition', [0, 0, 1, 1])
hold on
L = plot(ones(height(materials_master.('name'))), 'LineStyle', 'none', 'marker', 's', 'visible', 'on');
set(L, {'MarkerFaceColor'}, markerColor, {'MarkerEdgeColor'}, markerColor);
colormap(materials_master.('map'))
legend(materials_master.('name'))
end