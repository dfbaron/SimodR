bpmn_get_elements<-function (xml, model_ns = "http://www.omg.org/spec/BPMN/20100524/MODEL") 
{
  process <- xml2::xml_find_all(xml, "//model:process", c(model = model_ns))
  elements_xml <- xml2::xml_children(process)
  bpmn_elements <- data.frame(type = xml2::xml_name(elements_xml), 
                              name = xml2::xml_attr(elements_xml, attr = "name"), 
                              sourceRef = xml2::xml_attr(elements_xml, attr = "sourceRef"), 
                              targetRef = xml2::xml_attr(elements_xml, attr = "targetRef"), 
                              id = xml2::xml_attr(elements_xml, attr = "id"), stringsAsFactors = FALSE)
}