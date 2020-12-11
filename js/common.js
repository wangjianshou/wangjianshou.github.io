//左侧边栏控制函数

//获取元素到根元素顶部的距离
function getElePos(ele){
  var topPos = 0;
  while(ele){
    topPos += ele.offsetTop;
    ele = ele.offsetParent;
  }
  return topPos;
}

//建立目录
function buildContents(){
    var sideBarDiv = document.getElementById('sideBarDiv');
    var mainText = document.getElementById('mainText');
    var nodes = mainText.getElementsByTagName('*');
    var dlist = document.createElement('dl');
    dlist.setAttribute('id', 'ContentsID');
    sideBarDiv.appendChild(dlist);
    var num = 0; //计算进入目录的两级元素（h2，h3）个数
    for(var i=0; i < nodes.length; i++){
      if(nodes[i].nodeName=='H2' || nodes[i].nodeName=='H3'){
        nodes[i].id || nodes[i].setAttribute('id', 'title'+num.toString());
        var item;
        switch(nodes[i].nodeName){
          case 'H2':
            item = document.createElement('dt');
            break;
          case 'H3':
            item = document.createElement('dd');
            break;
        }
        item.setAttribute('name', nodes[i].id);
        item.appendChild(document.createTextNode(nodes[i].innerText));
        dlist.append(item);
        num++;
        item.onclick = function(){
          var pos = getElePos(document.getElementById(this.getAttribute('name')));
          window.scrollTo(0, pos-50);
        };
      } 
    } 
}   

var ScrollObj = new Object();
$(document).ready(function(){
  buildContents();
  ScrollObj.dlist = document.getElementById('ContentsID');
  ScrollObj.nodesPos = new Array();
  ScrollObj.nodes = ScrollObj.dlist.getElementsByTagName('*');
  for(let i=0; i<ScrollObj.nodes.length; i++){
     ScrollObj.nodesPos[i] = getElePos(document.getElementById(ScrollObj.nodes[i].getAttribute('name')));
  };
  ScrollObj.setCurrentNode = function(){
    let scrollBarPos = document.body.scrollTop || document.documentElement.scrollTop;
    scrollBarPos = scrollBarPos+50;
    for(var idx=1; idx < ScrollObj.nodes.length; idx++){
      //let h = document.getElementById(ScrollObj.nodes[idx-1].getAttribute('name')).offsetHeight;
      let dy = Math.min(ScrollObj.nodesPos[idx]-ScrollObj.nodesPos[idx-1], window.innerHeight);
      if(scrollBarPos < ScrollObj.nodesPos[idx]-0.5*dy){
        break;
      }
    };
    ScrollObj.nodes[idx-1].className="active";
    ScrollObj.currNode = ScrollObj.nodes[idx-1];
  }
  ScrollObj.setCurrentNode();
  window.onscroll = function(){
    ScrollObj.currNode.removeAttribute('class');
    ScrollObj.setCurrentNode();
    //$("#test").text(document.documentElement.scrollTop||document.body.scrollTop);
  }
});
