function onPageLoad(){
    console.log('document loaded');
    var url = 'http://127.0.0.1:5000/get_location_names';
    //var url = "/api/get_location_names";  // using nginx
    $.get(url, function(data, status){
        console.log('got response for get_location_names_request');
        if(data){
            var locations = data.location;
            var uiLocation = document.getElementById('uiLocation');
            $('#uiLocation').empty();
            for(var i in locations){
                var opt = new Option(locations[i]);
                $('#uiLocation').append(opt);
            }
        }
    });
}

function getBathValue(){
    var uiBATH = document.getElementById('uiBATH');
    for(var i in uiBATH){
        if(uiBATH[i].checked){
            return parseInt(i) + 1;  // i in loop start from 0
        }
    }
    return -1; // invalid value
}

function getBHKValue(){
    var uiBHK = document.getElementById('uiBHK');
    for(var i in uiBHK){
        if(uiBHK[i].checked){
            return parseInt(i) + 1; // i in loop start from 0
        }
    }
    return -1; // invalid value
}

function onClickedEstimatePrice(){
    console.log('button is clicked');
    var sqft = document.getElementById('uiSqft');
    var location = document.getElementById('uiLocation');
    var bhk = getBHKValue();
    var bath = getBathValue();
    var estPrice = document.getElementById('uiEstimatedPrice');

    var url = 'http://127.0.0.1:5000/predict_home_price';
    //var url = "/api/get_location_names";  // using nginx

    $.post(url, {
        total_sqft: parseFloat(sqft.value),
        bhk: bhk,
        bath: bath,
        location: location.value
    }, function(data, status){
        console.log(data.estimated_price);
        estPrice.innerHTML = "<h2>" + data.estimated_price.toString() + " </h2>";
        console.log(status);
    });

}

window.onload = onPageLoad;