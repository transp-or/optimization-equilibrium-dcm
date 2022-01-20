# PhD Thesis: Optimization and equilibrium problems with discrete choice models

The PhD thesis was developed at EPFL between November 2017 and November 2021 by [Stefano Bortolomiol], under the supervision of Michel Bierlaire and Virginie Lurkin.

## Content

This repository contains all the code related to the experiments contained in the PhD thesis. The thesis is composed of three core sections.

- Chapter 2, based on the following article: Bortolomiol, S., Lurkin, V., Bierlaire, M. (2021). [A simulation-based heuristic to find approximate equilibria with disaggregate demand models]. Transportation Science, 55(5):1025–1045.

- Chapter 3, based on the following article: Bortolomiol, S., Lurkin, V., Bierlaire, M. (2021). [Price-based regulation of oligopolistic markets under discrete choice models of demand]. Transportation.

- Chapter 4, which builds upon the work presented in the following conference paper: Bortolomiol, S., Lurkin, V., Bierlaire, M., Bongiovanni, C. (2021). [Benders decomposition for choice-based optimization problems with discrete upper-level variables]. In Proceedings of the 21st Swiss Transport Research Conference, Ascona, Switzerland.

_Software requirements: all algorithms are coded in Python, and all MILPs are solved using CPLEX. As of January 2022, CPLEX is available free of charge to all academic users through the IBM ILOG Optimization [Academic Initiative]._

There are five folders in the main repository. The following table outlines the relationship between the various case studies and the corresponding sections of the thesis:

| Case study | Thesis section |
| ------ | ------ |
| [case-study-Lin-Sibdari] | 2.5.1 |
| [case-study-parking] | 2.5.2 |
| [case-study-HSR] | 2.5.3 |
| [case-study-intercity-travel] | 3.4 |
| [benders-facility-location-pricing] | 4.2 + 4.6 |


## License

[MIT License]

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. See StackOverflow: http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Stefano Bortolomiol]: <https://www.linkedin.com/in/stefano-bortolomiol/>

   [A simulation-based heuristic to find approximate equilibria with disaggregate demand models]: <https://pubsonline.informs.org/doi/abs/10.1287/trsc.2021.1071>
   [Price-based regulation of oligopolistic markets under discrete choice models of demand]: <https://link.springer.com/article/10.1007/s11116-021-10217-0>
   [Benders decomposition for choice-based optimization problems with discrete upper-level variables]: <http://strc.ch/2021/Bortolomiol_EtAl.pdf>
   
   [Academic Initiative]: <https://content-eu-7.content-cms.com/b73a5759-c6a6-4033-ab6b-d9d4f9a6d65b/dxsites/151914d1-03d2-48fe-97d9-d21166848e65/academic/home>
   
   [case-study-Lin-Sibdari]: <https://github.com/stefanoborto/optimization-equilibrium-dcm/tree/main/case-study-Lin-Sibdari>
   [case-study-parking]: <https://github.com/stefanoborto/optimization-equilibrium-dcm/tree/main/case-study-parking>
   [case-study-HSR]: <https://github.com/stefanoborto/optimization-equilibrium-dcm/tree/main/case-study-HSR>
   [case-study-intercity-travel]: <https://github.com/stefanoborto/optimization-equilibrium-dcm/tree/main/case-study-intercity-travel>
   [benders-facility-location-pricing]: <https://github.com/stefanoborto/optimization-equilibrium-dcm/tree/main/benders-facility-location-pricing>
   
   [MIT License]: <https://opensource.org/licenses/MIT>
  
